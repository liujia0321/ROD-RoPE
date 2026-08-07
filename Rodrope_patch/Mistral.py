# transfromers version 4.38.2
# No support of sliding window. Check our paper for more reason about why we don't use it.
import torch
import torch.nn as nn
import math
import warnings
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import numpy as np
from flash_attn import flash_attn_func, flash_attn_varlen_func

from .Rodrope_flash_attn import rodrope_flash_forward



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos[:,:, -q.shape[2]:]) + (rotate_half(q) * sin[:,:, -q.shape[2]:]) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed

def apply_grouped_rotary_pos_emb(q, k, cos, sin, position_ids, g_size_1=1, g_size_2=4096):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    position_ids_q = position_ids//g_size_1 + g_size_2 - g_size_2//g_size_1
    position_ids_k = position_ids//g_size_1

    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos_q = cos[position_ids_q].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_q = sin[position_ids_q].unsqueeze(1)  # [bs, 1, seq_len, dim]
    cos_k = cos[position_ids_k].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_k = sin[position_ids_k].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q) if q is not None else None
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k) if k is not None else None

    return q_embed, k_embed


def _manual_rotary_emb_for_positions(rotary_emb, value_states, position_ids):
    """Build cos/sin for arbitrary integer or compressed float positions."""
    inv_freq = rotary_emb.inv_freq.to(device=position_ids.device)
    pos = position_ids.to(dtype=inv_freq.dtype)
    freqs = torch.einsum("...i,j->...ij", pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=value_states.dtype)
    sin = emb.sin().to(dtype=value_states.dtype)
    return cos, sin


def rotary_emb_for_positions(self, value_states, position_ids):
    """Return cos/sin already selected for position_ids as [bs, seq, dim]."""
    try:
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        if cos.dim() == 4:
            cos = cos.squeeze(1).squeeze(0)
            sin = sin.squeeze(1).squeeze(0)
            if cos.dim() == 2:
                cos = cos[position_ids.long()]
                sin = sin[position_ids.long()]
        return cos.to(value_states.dtype), sin.to(value_states.dtype)
    except TypeError:
        return _manual_rotary_emb_for_positions(self.rotary_emb, value_states, position_ids)


def apply_selected_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply cos/sin tensors that are already selected for each token position."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed


def rodrope_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 2048,
    scale_base: Optional[float] = -1,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
    query_position = position_ids
    # only consider bsz=1 for now. 
    key_position = torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)


    neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position) 
    _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, cos, sin, key_position) 
    _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
    group_query_states, _ = apply_grouped_rotary_pos_emb(scaled_query, None, cos, sin, query_position, g_size_1=group_size_1, g_size_2=_re_group_size_2) 
    _, group_key_states = apply_grouped_rotary_pos_emb(None, key_states, cos, sin, key_position, g_size_1=group_size_1, g_size_2=_re_group_size_2) 


    group_key_states = repeat_kv(group_key_states, self.num_key_value_groups)
    neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 


    if group_attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {group_attn_weights.size()}"
        )
    
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        group_attn_weights = group_attn_weights + attention_mask
        neighbor_attn_weights = neighbor_attn_weights + attention_mask


    if q_len == 1:
        neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask[:, -group_size_2:] = 1
    elif q_len == kv_seq_len:
        neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        if q_len-group_size_2 > 0:
            group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
            neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask

    else:
        raise ValueError("q_len should be 1 or seq_len.")


    neighbor_attention_mask = neighbor_attention_mask.bool()
    attn_weights = torch.where(neighbor_attention_mask, neighbor_attn_weights, group_attn_weights)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def flash_rodrope_two_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 2048,
    scale_base: Optional[float] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype)
    else:
        scaled_query = query_states

    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)
    neighbor_window = int(group_size_2)
    attn_dropout = self.config.attention_dropout if self.training else 0.0

    if q_len == 1:
        current_position = position_ids[:, -1:]
        neighbor_key_position = current_position - key_position
        _re_group_size_2 = 0 if position_ids.max() < neighbor_window else neighbor_window
        group_key_position = (
            current_position // group_size_1
            - key_position // group_size_1
            + (_re_group_size_2 - _re_group_size_2 / group_size_1)
        )
        neighbor_start = max(kv_seq_len - neighbor_window, 0)
        decode_key_position = torch.cat(
            [group_key_position[:, :neighbor_start], neighbor_key_position[:, neighbor_start:]],
            dim=1,
        )

        decode_k_cos, decode_k_sin = rotary_emb_for_positions(self, value_states, decode_key_position)
        decode_query_states = scaled_query.transpose(1, 2).contiguous()
        _, decode_key_states = apply_selected_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin)

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        attn_output = flash_attn_func(
            decode_query_states,
            decode_key_states,
            decode_value_states,
            attn_dropout,
            softmax_scale=None,
            causal=True,
        )
    elif q_len == kv_seq_len:
        neighbor_q_cos, neighbor_q_sin = rotary_emb_for_positions(self, value_states, query_position)
        neighbor_k_cos, neighbor_k_sin = rotary_emb_for_positions(self, value_states, key_position)

        _re_group_size_2 = 0 if query_position.max() < neighbor_window else neighbor_window
        group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1

        group_q_cos, group_q_sin = rotary_emb_for_positions(self, value_states, group_query_position)
        group_k_cos, group_k_sin = rotary_emb_for_positions(self, value_states, group_key_position)

        neighbor_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin)
        _, neighbor_key_states = apply_selected_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin)
        group_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin)
        _, group_key_states = apply_selected_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin)

        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        attn_output = rodrope_flash_forward(
            self,
            query_position,
            neighbor_window,
            neighbor_query_states,
            neighbor_key_states,
            group_query_states,
            group_key_states,
            value_states,
            attention_mask,
            bsz,
            q_len,
            kv_seq_len,
            attn_dropout,
        )
    else:
        raise ValueError("q_len should be 1 or seq_len.")

    attn_output = attn_output.contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None if not output_attentions else None
    return attn_output, attn_weights, past_key_value


def flash_rodrope_three_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 2048,
    far_size: Optional[int] = None,
    far_group_size: Optional[float] = None,
    block: Optional[int] = 3,
    scale_base: Optional[float] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    if block != 3:
        raise ValueError("flash_rodrope_three_block_forward requires block=3.")
    if far_size is None or far_group_size is None:
        raise ValueError("block=3 requires far_size and far_group_size.")

    neighbor_window = int(group_size_2)
    far_size = int(far_size)
    far_group_size = float(far_group_size)
    use_far = far_size > neighbor_window and far_group_size > 0

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype)
    else:
        scaled_query = query_states

    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)
    attn_dropout = self.config.attention_dropout if self.training else 0.0

    if q_len == 1:
        current_position = position_ids[:, -1:]
        neighbor_key_position = current_position - key_position
        _re_group_size_2 = 0 if position_ids.max() < neighbor_window else neighbor_window
        group_key_position = (
            current_position // group_size_1
            - key_position // group_size_1
            + (_re_group_size_2 - _re_group_size_2 / group_size_1)
        )
        neighbor_start = max(kv_seq_len - neighbor_window, 0)
        if use_far and position_ids.max() >= far_size:
            far_offset = neighbor_window + (far_size - neighbor_window) / group_size_1 - far_size / far_group_size
            far_key_position = current_position // far_group_size - key_position // far_group_size + far_offset
            far_end = max(kv_seq_len - far_size, 0)
            decode_key_position = torch.cat(
                [
                    far_key_position[:, :far_end],
                    group_key_position[:, far_end:neighbor_start],
                    neighbor_key_position[:, neighbor_start:],
                ],
                dim=1,
            )
        else:
            decode_key_position = torch.cat(
                [group_key_position[:, :neighbor_start], neighbor_key_position[:, neighbor_start:]],
                dim=1,
            )

        decode_k_cos, decode_k_sin = rotary_emb_for_positions(self, value_states, decode_key_position)
        decode_query_states = scaled_query.transpose(1, 2).contiguous()
        _, decode_key_states = apply_selected_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin)

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        attn_output = flash_attn_func(
            decode_query_states,
            decode_key_states,
            decode_value_states,
            attn_dropout,
            softmax_scale=None,
            causal=True,
        )
    elif q_len == kv_seq_len:
        neighbor_q_cos, neighbor_q_sin = rotary_emb_for_positions(self, value_states, query_position)
        neighbor_k_cos, neighbor_k_sin = rotary_emb_for_positions(self, value_states, key_position)

        _re_group_size_2 = 0 if query_position.max() < neighbor_window else neighbor_window
        group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1
        group_q_cos, group_q_sin = rotary_emb_for_positions(self, value_states, group_query_position)
        group_k_cos, group_k_sin = rotary_emb_for_positions(self, value_states, group_key_position)

        neighbor_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin)
        _, neighbor_key_states = apply_selected_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin)
        group_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin)
        _, group_key_states = apply_selected_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin)

        far_query_states = None
        far_key_states = None
        if use_far and query_position.max() >= far_size:
            far_offset = neighbor_window + (far_size - neighbor_window) / group_size_1 - far_size / far_group_size
            far_query_position = query_position // far_group_size + far_offset
            far_key_position = key_position // far_group_size
            far_q_cos, far_q_sin = rotary_emb_for_positions(self, value_states, far_query_position)
            far_k_cos, far_k_sin = rotary_emb_for_positions(self, value_states, far_key_position)
            far_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, far_q_cos, far_q_sin)
            _, far_key_states = apply_selected_rotary_pos_emb(None, key_states, far_k_cos, far_k_sin)

        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        if far_query_states is not None:
            far_query_states = far_query_states.transpose(1, 2).contiguous()
            far_key_states = repeat_kv(far_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        attn_output = rodrope_flash_forward(
            self,
            query_position,
            neighbor_window,
            neighbor_query_states,
            neighbor_key_states,
            group_query_states,
            group_key_states,
            value_states,
            attention_mask,
            bsz,
            q_len,
            kv_seq_len,
            attn_dropout,
            far_size=far_size if use_far else None,
            far_query_states=far_query_states,
            far_key_states=far_key_states,
        )
    else:
        raise ValueError("q_len should be 1 or seq_len.")

    attn_output = attn_output.contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None if not output_attentions else None
    return attn_output, attn_weights, past_key_value



def _rodrope_flash_forward_four_block(
        model_self,
        query_position,
        m1,
        m2,
        m3,
        neighbor_query_states,
        neighbor_key_states,
        group_query_states,
        group_key_states,
        far_query_states,
        far_key_states,
        far2_query_states,
        far2_key_states,
        value_states,
        attention_mask,
        bsz,
        q_len,
        kv_seq_len,
        attn_dropout,
    ):

    m1 = int(m1)
    m2 = int(m2)
    m3 = int(m3)

    def _right_align_lse(lse_right_padded, seq_length):
        softmax_lse = torch.full_like(lse_right_padded, -float("inf"))
        for idx in range(bsz):
            length = int(seq_length[idx].item())
            if length > 0:
                softmax_lse[idx, :, -length:] = lse_right_padded[idx, :, :length]
        return softmax_lse

    def _seq_length(seq_len, mask):
        return (
            torch.full((bsz, 1), seq_len, dtype=torch.long, device=query_position.device)
            if mask is None
            else torch.sum(mask, axis=1, keepdim=True)
        )

    def _pad_output_tail(local_output, total_len):
        padded = torch.zeros(
            (local_output.shape[0], total_len, local_output.shape[2], local_output.shape[3]),
            dtype=local_output.dtype,
            device=local_output.device,
        )
        padded[:, -local_output.shape[1]:, ...] = local_output
        return padded

    def _pad_lse_tail(local_lse, total_len):
        padded = torch.full(
            (local_lse.shape[0], total_len, local_lse.shape[2], local_lse.shape[3]),
            -float("inf"),
            dtype=local_lse.dtype,
            device=local_lse.device,
        )
        padded[:, -local_lse.shape[1]:, ...] = local_lse
        return padded

    def _align_lse(lse_right_padded, seq_length):
        return _right_align_lse(lse_right_padded, seq_length).transpose(1, 2).unsqueeze(-1)

    if query_position.max() >= m1:
        neighbor_attn_output, neighbor_softmax_lse_right_padded, _ = model_self._flash_attention_forward(
            neighbor_query_states,
            neighbor_key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            window_size=[m1 - 1, 0],
            return_attn_probs=True,
        )

        output_list = [neighbor_attn_output]
        lse_list = [
            _align_lse(
                neighbor_softmax_lse_right_padded,
                _seq_length(kv_seq_len, attention_mask),
            )
        ]

        group_attention_len = kv_seq_len - m1
        group_attention_mask = attention_mask[:, :group_attention_len] if attention_mask is not None else None
        group_attn_output, group_softmax_lse_right_padded, _ = model_self._flash_attention_forward(
            group_query_states[:, -group_attention_len:, :, :],
            group_key_states[:, :group_attention_len, :, :],
            value_states[:, :group_attention_len, :, :],
            group_attention_mask,
            group_query_states[:, -group_attention_len:, :, :].shape[1],
            dropout=attn_dropout,
            window_size=[m2 - m1 - 1, 0],
            return_attn_probs=True,
        )
        output_list.append(_pad_output_tail(group_attn_output, q_len))
        lse_list.append(
            _pad_lse_tail(
                _align_lse(
                    group_softmax_lse_right_padded,
                    _seq_length(group_attention_len, group_attention_mask),
                ),
                q_len,
            )
        )

        if query_position.max() >= m2:
            far_attention_len = kv_seq_len - m2
            far_attention_mask = attention_mask[:, :far_attention_len] if attention_mask is not None else None
            far_attn_output, far_softmax_lse_right_padded, _ = model_self._flash_attention_forward(
                far_query_states[:, -far_attention_len:, :, :],
                far_key_states[:, :far_attention_len, :, :],
                value_states[:, :far_attention_len, :, :],
                far_attention_mask,
                far_query_states[:, -far_attention_len:, :, :].shape[1],
                dropout=attn_dropout,
                window_size=[m3 - m2 - 1, 0],
                return_attn_probs=True,
            )
            output_list.append(_pad_output_tail(far_attn_output, q_len))
            lse_list.append(
                _pad_lse_tail(
                    _align_lse(
                        far_softmax_lse_right_padded,
                        _seq_length(far_attention_len, far_attention_mask),
                    ),
                    q_len,
                )
            )

        if query_position.max() >= m3:
            far2_attention_len = kv_seq_len - m3
            far2_attention_mask = attention_mask[:, :far2_attention_len] if attention_mask is not None else None
            far2_attn_output, far2_softmax_lse_right_padded, _ = model_self._flash_attention_forward(
                far2_query_states[:, -far2_attention_len:, :, :],
                far2_key_states[:, :far2_attention_len, :, :],
                value_states[:, :far2_attention_len, :, :],
                far2_attention_mask,
                far2_query_states[:, -far2_attention_len:, :, :].shape[1],
                dropout=attn_dropout,
                window_size=[-1, -1],
                return_attn_probs=True,
            )
            output_list.append(_pad_output_tail(far2_attn_output, q_len))
            lse_list.append(
                _pad_lse_tail(
                    _align_lse(
                        far2_softmax_lse_right_padded,
                        _seq_length(far2_attention_len, far2_attention_mask),
                    ),
                    q_len,
                )
            )

        output_stack = torch.stack(output_list, dim=0)
        lse_stack = torch.stack(lse_list, dim=0)
        weight_stack = torch.softmax(lse_stack, dim=0)
        attn_output = torch.sum(output_stack * weight_stack, dim=0)
        attn_output = torch.nan_to_num(attn_output, nan=0).to(neighbor_attn_output.dtype)

    else:
        attn_output = model_self._flash_attention_forward(
            neighbor_query_states,
            neighbor_key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            window_size=[-1, -1],
        )

    return attn_output


def flash_rodrope_four_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 2048,
    far_size: Optional[int] = None,
    far_group_size: Optional[float] = None,
    far2_size: Optional[int] = None,
    far2_group_size: Optional[float] = None,
    block: Optional[int] = 4,
    scale_base: Optional[float] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Four-block Mistral ROD-RoPE flash attention:
    neighbor/original, group(lambda2), far(lambda3), far2(lambda4).
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    if block != 4:
        raise ValueError("flash_rodrope_four_block_forward requires block=4.")
    if far_size is None or far_group_size is None or far2_size is None or far2_group_size is None:
        raise ValueError("block=4 requires far_size, far_group_size, far2_size and far2_group_size.")

    neighbor_window = int(group_size_2)
    far_size = int(far_size)
    far_group_size = float(far_group_size)
    far2_size = int(far2_size)
    far2_group_size = float(far2_group_size)
    if far_size <= neighbor_window:
        raise ValueError("block=4 requires far_size > group_size_2.")
    if far2_size <= far_size:
        raise ValueError("block=4 requires far2_size > far_size.")
    if far_group_size <= 0 or far2_group_size <= 0:
        raise ValueError("block=4 requires positive far_group_size and far2_group_size.")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype)
    else:
        scaled_query = query_states

    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)
    attn_dropout = self.config.attention_dropout if self.training else 0.0

    if q_len == 1:
        current_position = position_ids[:, -1:]
        neighbor_key_position = current_position - key_position

        _re_group_size_2 = 0 if position_ids.max() < neighbor_window else neighbor_window
        group_key_position = (
            current_position // group_size_1
            - key_position // group_size_1
            + (_re_group_size_2 - _re_group_size_2 / group_size_1)
        )

        far_offset = neighbor_window + (far_size - neighbor_window) / group_size_1 - far_size / far_group_size
        far_key_position = current_position // far_group_size - key_position // far_group_size + far_offset
        far2_offset = (
            neighbor_window
            + (far_size - neighbor_window) / group_size_1
            + (far2_size - far_size) / far_group_size
            - far2_size / far2_group_size
        )
        far2_key_position = current_position // far2_group_size - key_position // far2_group_size + far2_offset

        neighbor_start = max(kv_seq_len - neighbor_window, 0)
        if position_ids.max() >= far2_size:
            far2_end = max(kv_seq_len - far2_size, 0)
            far_end = max(kv_seq_len - far_size, 0)
            decode_key_position = torch.cat(
                [
                    far2_key_position[:, :far2_end],
                    far_key_position[:, far2_end:far_end],
                    group_key_position[:, far_end:neighbor_start],
                    neighbor_key_position[:, neighbor_start:],
                ],
                dim=1,
            )
        elif position_ids.max() >= far_size:
            far_end = max(kv_seq_len - far_size, 0)
            decode_key_position = torch.cat(
                [
                    far_key_position[:, :far_end],
                    group_key_position[:, far_end:neighbor_start],
                    neighbor_key_position[:, neighbor_start:],
                ],
                dim=1,
            )
        else:
            decode_key_position = torch.cat(
                [group_key_position[:, :neighbor_start], neighbor_key_position[:, neighbor_start:]],
                dim=1,
            )

        decode_k_cos, decode_k_sin = rotary_emb_for_positions(self, value_states, decode_key_position)
        decode_query_states = scaled_query.transpose(1, 2).contiguous()
        _, decode_key_states = apply_selected_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin)

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        attn_output = flash_attn_func(
            decode_query_states,
            decode_key_states,
            decode_value_states,
            attn_dropout,
            softmax_scale=None,
            causal=True,
        )
    elif q_len == kv_seq_len:
        neighbor_q_cos, neighbor_q_sin = rotary_emb_for_positions(self, value_states, query_position)
        neighbor_k_cos, neighbor_k_sin = rotary_emb_for_positions(self, value_states, key_position)

        _re_group_size_2 = 0 if query_position.max() < neighbor_window else neighbor_window
        group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1

        far_offset = neighbor_window + (far_size - neighbor_window) / group_size_1 - far_size / far_group_size
        far_query_position = query_position // far_group_size + far_offset
        far_key_position = key_position // far_group_size

        far2_offset = (
            neighbor_window
            + (far_size - neighbor_window) / group_size_1
            + (far2_size - far_size) / far_group_size
            - far2_size / far2_group_size
        )
        far2_query_position = query_position // far2_group_size + far2_offset
        far2_key_position = key_position // far2_group_size

        group_q_cos, group_q_sin = rotary_emb_for_positions(self, value_states, group_query_position)
        group_k_cos, group_k_sin = rotary_emb_for_positions(self, value_states, group_key_position)
        far_q_cos, far_q_sin = rotary_emb_for_positions(self, value_states, far_query_position)
        far_k_cos, far_k_sin = rotary_emb_for_positions(self, value_states, far_key_position)
        far2_q_cos, far2_q_sin = rotary_emb_for_positions(self, value_states, far2_query_position)
        far2_k_cos, far2_k_sin = rotary_emb_for_positions(self, value_states, far2_key_position)

        neighbor_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin)
        _, neighbor_key_states = apply_selected_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin)
        group_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin)
        _, group_key_states = apply_selected_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin)
        far_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, far_q_cos, far_q_sin)
        _, far_key_states = apply_selected_rotary_pos_emb(None, key_states, far_k_cos, far_k_sin)
        far2_query_states, _ = apply_selected_rotary_pos_emb(scaled_query, None, far2_q_cos, far2_q_sin)
        _, far2_key_states = apply_selected_rotary_pos_emb(None, key_states, far2_k_cos, far2_k_sin)

        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        far_query_states = far_query_states.transpose(1, 2).contiguous()
        far_key_states = repeat_kv(far_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        far2_query_states = far2_query_states.transpose(1, 2).contiguous()
        far2_key_states = repeat_kv(far2_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        attn_output = _rodrope_flash_forward_four_block(
            self,
            query_position,
            neighbor_window,
            far_size,
            far2_size,
            neighbor_query_states,
            neighbor_key_states,
            group_query_states,
            group_key_states,
            far_query_states,
            far_key_states,
            far2_query_states,
            far2_key_states,
            value_states,
            attention_mask,
            bsz,
            q_len,
            kv_seq_len,
            attn_dropout,
        )
    else:
        raise ValueError("q_len should be 1 or seq_len.")

    attn_output = attn_output.contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    attn_weights = None if not output_attentions else None
    return attn_output, attn_weights, past_key_value

def flash_rodrope_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 2048,
    far_size: Optional[int] = None,
    far_group_size: Optional[float] = None,
    far2_size: Optional[int] = None,
    far2_group_size: Optional[float] = None,
    block: Optional[int] = 2,
    scale_base: Optional[float] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if block == 2:
        return flash_rodrope_two_block_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            group_size_1=group_size_1,
            group_size_2=group_size_2,
            scale_base=scale_base,
            cache_position=cache_position,
            **kwargs,
        )
    if block == 3:
        return flash_rodrope_three_block_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            group_size_1=group_size_1,
            group_size_2=group_size_2,
            far_size=far_size,
            far_group_size=far_group_size,
            block=block,
            scale_base=scale_base,
            cache_position=cache_position,
            **kwargs,
        )
    if block == 4:
        return flash_rodrope_four_block_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            group_size_1=group_size_1,
            group_size_2=group_size_2,
            far_size=far_size,
            far_group_size=far_group_size,
            far2_size=far2_size,
            far2_group_size=far2_group_size,
            block=block,
            scale_base=scale_base,
            cache_position=cache_position,
            **kwargs,
        )
    raise ValueError("block should be 2 for original Rodrope, 3 for three-block Rodrope or 4 for four-block Rodrope.")

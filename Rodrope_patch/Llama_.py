import torch
from transformers.models.llama.modeling_llama import *
from transformers.models.gpt_neox.modeling_gpt_neox import *
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from transformers.cache_utils import Cache
from flash_attn import flash_attn_func, flash_attn_varlen_func
from .Rodrope_flash_attn import rodrope_flash_forward, rodrope_flash_forward_two_block
from .Rodrope_flash_attn_triton import rodrope_flash_forward_triton



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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) if not q is None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if not k is None else None
    return q_embed, k_embed



def flash_rodrope_two_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    scale_base: Optional[int] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        Require updating tansformers to >= 4.38.2, flash_attn >= 2.5.6
        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Never test it with batch size > 1.
        d. Only support q_len = 1 or q_len = seq_len.
    """
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
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale 
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale 
    else:
        scaled_query = query_states
    
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    # only consider bsz=1 for now. 
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len) 
    attn_dropout = self.config.attention_dropout if self.training else 0.0
    if q_len == 1:
        # We implement the case q_len == 1 separately, by manipulating positions.
        # for our flash implementation doesnot work for  decoding stage at the releasing time.

        neighbor_key_position = position_ids[:, -1] - key_position
        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2
        group_key_position = position_ids[:, -1]//group_size_1 - key_position//group_size_1 #+ (_re_group_size_2 - _re_group_size_2//group_size_1)
        decode_key_position = torch.cat([group_key_position[:, :-group_size_2], neighbor_key_position[:,-group_size_2:]], dim=1)
        
        decode_k_cos, decode_k_sin = self.rotary_emb(value_states, decode_key_position, seq_len=None)
        #import pdb; pdb.set_trace()
        #neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, cos, sin, query_position_ids) 
        decode_query_states = scaled_query.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
        _, decode_key_states = apply_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin, decode_key_position) 

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        
        attn_output = flash_attn_func(decode_query_states,
                                      decode_key_states,
                                      decode_value_states,
                                      attn_dropout, 
                                      softmax_scale=None, 
                                      causal=True)
    elif q_len == kv_seq_len:
        # set correct position_ids & apply RoPE.
        
        neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position, seq_len=None)
        neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position, seq_len=None)

        _re_group_size_2 = 0 if query_position.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
        group_query_position = query_position // group_size_1 #+ _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1

        group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position, seq_len=None)
        group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position, seq_len=None)

        neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
        _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
        group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
        _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)
        

        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        attn_output = rodrope_flash_forward_two_block(self,
                                                query_position,
                                                group_size_2,
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
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def flash_rodrope_three_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    far_size: Optional[int] = None,
    far_group_size: Optional[float] = None,
    block: Optional[int] = 3,
    scale_base: Optional[int] = -1,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
        Require updating tansformers to >= 4.38.2, flash_attn >= 2.5.6
        a. Only support causal mask.
        b. Don't support atttention_mask.
        c. Never test it with batch size > 1.
        d. Only support q_len = 1 or q_len = seq_len.

        block=2: original Rodrope with neighbor/group blocks.
        block=3: three-block mode:
        - neighbor: distance < group_size_2, original RoPE.
        - group: group_size_2 <= distance < far_size, compressed by group_size_1.
        - far: distance >= far_size, compressed by far_group_size.
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop("padding_mask")

    if block != 3:
        raise ValueError("flash_rodrope_three_block_forward requires block=3.")

    neighbor_window = int(group_size_2)
    if far_size is None or far_group_size is None:
        raise ValueError("block=3 requires far_size and far_group_size.")

    use_far = far_size is not None and far_group_size is not None
    if use_far:
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
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype) # log scale
        #scaled_query = query_states * (((0.1*(((position_ids+1)[:, None, :, None]/scale_base).log())+1)**2).clip(1)).to(query_states.dtype) # Yarn scale
    else:
        scaled_query = query_states

    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; position_ids needed for the static cache
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    kv_seq_len = key_states.shape[-2]

    query_position = position_ids
    # only consider bsz=1 for now.
    key_position = position_ids if q_len != 1 else torch.arange(kv_seq_len, dtype=position_ids.dtype).to(query_position.device).view(1, kv_seq_len)
    attn_dropout = self.config.attention_dropout if self.training else 0.0
    if q_len == 1:
        # We implement the decoding case by directly assigning each cached key its effective relative RoPE distance.
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

        decode_k_cos, decode_k_sin = self.rotary_emb(value_states, decode_key_position, seq_len=None)
        decode_query_states = scaled_query.transpose(1,2).contiguous() # position 0: cos 0 = 1, sin 0 = 0
        _, decode_key_states = apply_rotary_pos_emb(None, key_states, decode_k_cos, -decode_k_sin, decode_key_position)

        decode_key_states = repeat_kv(decode_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        decode_value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        attn_output = flash_attn_func(decode_query_states,
                                      decode_key_states,
                                      decode_value_states,
                                      attn_dropout,
                                      softmax_scale=None,
                                      causal=True)
    elif q_len == kv_seq_len:
        # set correct position_ids & apply RoPE.
        neighbor_q_cos, neighbor_q_sin = self.rotary_emb(value_states, query_position, seq_len=None)
        neighbor_k_cos, neighbor_k_sin = self.rotary_emb(value_states, key_position, seq_len=None)

        _re_group_size_2 = 0 if query_position.max() < neighbor_window else neighbor_window # in case that, the smallest q position, g2-g2//g1 exceed the max position
        group_query_position = query_position // group_size_1 + _re_group_size_2 - _re_group_size_2 / group_size_1
        group_key_position = key_position // group_size_1

        group_q_cos, group_q_sin = self.rotary_emb(value_states, group_query_position, seq_len=None)
        group_k_cos, group_k_sin = self.rotary_emb(value_states, group_key_position, seq_len=None)

        neighbor_query_states, _ = apply_rotary_pos_emb(scaled_query, None, neighbor_q_cos, neighbor_q_sin, None)
        _, neighbor_key_states = apply_rotary_pos_emb(None, key_states, neighbor_k_cos, neighbor_k_sin, None)
        group_query_states, _ = apply_rotary_pos_emb(scaled_query, None, group_q_cos, group_q_sin, None)
        _, group_key_states = apply_rotary_pos_emb(None, key_states, group_k_cos, group_k_sin, None)

        far_query_states = None
        far_key_states = None
        if use_far and query_position.max() >= far_size:
            far_offset = neighbor_window + (far_size - neighbor_window) / group_size_1 - far_size / far_group_size
            far_query_position = query_position // far_group_size + far_offset
            far_key_position = key_position // far_group_size
            far_q_cos, far_q_sin = self.rotary_emb(value_states, far_query_position, seq_len=None)
            far_k_cos, far_k_sin = self.rotary_emb(value_states, far_key_position, seq_len=None)
            far_query_states, _ = apply_rotary_pos_emb(scaled_query, None, far_q_cos, far_q_sin, None)
            _, far_key_states = apply_rotary_pos_emb(None, key_states, far_k_cos, far_k_sin, None)

        neighbor_query_states = neighbor_query_states.transpose(1, 2).contiguous()
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        group_query_states = group_query_states.transpose(1, 2).contiguous()
        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        if far_query_states is not None:
            far_query_states = far_query_states.transpose(1, 2).contiguous()
            far_key_states = repeat_kv(far_key_states, self.num_key_value_groups).transpose(1, 2).contiguous()
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2).contiguous()

        attn_output = rodrope_flash_forward(self,
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
    attn_output = attn_output.view(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    attn_weights = None
    return attn_output, attn_weights, past_key_value

def flash_rodrope_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 8,
    group_size_2: Optional[float] = 1024,
    far_size: Optional[int] = None,
    far_group_size: Optional[float] = None,
    block: Optional[int] = 2,
    scale_base: Optional[int] = -1,
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
            group_size_1=group_size_1,
            group_size_2=group_size_2,
            far_size=far_size,
            far_group_size=far_group_size,
            block=block,
            scale_base=scale_base,
            cache_position=cache_position,
            **kwargs,
        )
    raise ValueError("block should be 2 for original Rodrope or 3 for three-block Rodrope.")


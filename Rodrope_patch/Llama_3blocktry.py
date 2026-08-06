import math
import torch
import numpy as np
from typing import Optional, Tuple, List
from transformers.cache_utils import Cache
from flash_attn import flash_attn_func


# ------------------------- utils ------------------------- #
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
    k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed


def _linear_map_pos(pos: torch.LongTensor, a: float, b: float, clamp_min: int = 0):
    pos_f = pos.to(torch.float32)
    mapped = torch.floor(a * pos_f + b).to(torch.long)
    mapped = torch.clamp(mapped, min=clamp_min)
    return mapped


def flash_rodrope_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,  # only causal
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    scale_base: int = -1,
    # ----- 4-block settings -----
    w1: int = 1024,
    w2: int = 4096,
    w3: int = 8192,
    a_list: Optional[List[float]] = None,  # len=4
    b_list: Optional[List[float]] = None,  # len=4
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Runnable 4-block flash baseline WITHOUT LSE-merge (because your flash_attn_func doesn't support it).
    - Decode (q_len==1): 4-block segmentation + flash per segment + weighted-avg merge (approx).
    - Prefill (q_len==kv_seq_len): use only block1 (standard RoPE) + one flash call (fast & stable).

    Constraints:
      - causal only (attention_mask not supported)
      - bsz == 1
      - q_len == 1 or q_len == kv_seq_len
    """

    if attention_mask is not None:
        raise ValueError("Only causal is supported; attention_mask is not supported in this flash path.")

    bsz, q_len, _ = hidden_states.size()
    if bsz != 1:
        raise ValueError("This baseline only supports bsz=1 for now.")

    if a_list is None:
        a_list = [1.0, 0.5, 0.25, 0.125]
    if b_list is None:
        # for w1=1024,w2=4096,w3=8192:
        b_list = [0.0, 512.0, 1536.0, 2560.0]
    if len(a_list) != 4 or len(b_list) != 4:
        raise ValueError("a_list and b_list must have length 4.")

    # ----- QKV proj -----
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)          # [1,H,q,d]
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [1,Hkv,q,d]
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # optional query scaling
    if scale_base > 0:
        scaled_query = query_states * ((position_ids + 1)[:, None, :, None].log() / np.log(scale_base)).clip(1).to(query_states.dtype)
    else:
        scaled_query = query_states

    # cache update
    past_key_value = getattr(self, "past_key_value", past_key_value)
    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    kv_seq_len = key_states.shape[-2]
    if q_len not in (1, kv_seq_len):
        raise ValueError("q_len should be 1 (decode) or q_len == kv_seq_len (prefill).")

    query_position = position_ids  # [1,q]
    if q_len == 1:
        key_position = torch.arange(kv_seq_len, dtype=position_ids.dtype, device=position_ids.device).view(1, kv_seq_len)
    else:
        key_position = position_ids

    # expand V once
    value_states_full = repeat_kv(value_states, self.num_key_value_groups)  # [1,H,kv,d]
    attn_dropout = self.config.attention_dropout if self.training else 0.0

    # ----- helper: build block q/k (RoPE with mapped positions) -----
    def make_block_qk(block_idx: int):
        a = float(a_list[block_idx])
        b = float(b_list[block_idx])

        q_pos_map = _linear_map_pos(query_position, a=a, b=b)
        k_pos_map = _linear_map_pos(key_position,   a=a, b=b)

        q_cos, q_sin = self.rotary_emb(value_states, q_pos_map, seq_len=None)  # [1,q,hd]
        k_cos, k_sin = self.rotary_emb(value_states, k_pos_map, seq_len=None)  # [1,kv,hd]

        q_blk, _ = apply_rotary_pos_emb(scaled_query, None, q_cos, q_sin, unsqueeze_dim=1)
        _, k_blk = apply_rotary_pos_emb(None, key_states,  k_cos, k_sin, unsqueeze_dim=1)

        k_blk = repeat_kv(k_blk, self.num_key_value_groups)  # [1,H,kv,d]
        return q_blk, k_blk

    # precompute 4 blocks
    q1, k1 = make_block_qk(0)
    q2, k2 = make_block_qk(1)
    q3, k3 = make_block_qk(2)
    q4, k4 = make_block_qk(3)

    # =========================
    # Decode: q_len == 1 (4-block + flash per segment, approx merge)
    # =========================
    if q_len == 1:
        t = kv_seq_len - 1
        s1 = max(0, t - w1 + 1)
        s2 = max(0, t - w2 + 1)
        s3 = max(0, t - w3 + 1)

        segments = []
        if s3 > 0:
            segments.append(("b4", q4, k4, 0, s3))
        if s2 > s3:
            segments.append(("b3", q3, k3, s3, s2))
        if s1 > s2:
            segments.append(("b2", q2, k2, s2, s1))
        segments.append(("b1", q1, k1, s1, t + 1))

        merged_o = None
        wsum = 0.0

        for _, q_blk, k_blk, ks, ke in segments:
            q_seg = q_blk.transpose(1, 2).contiguous()                          # [1,1,H,d]
            k_seg = k_blk[:, :, ks:ke, :].transpose(1, 2).contiguous()          # [1,K,H,d]
            v_seg = value_states_full[:, :, ks:ke, :].transpose(1, 2).contiguous()

            o = flash_attn_func(
                q_seg, k_seg, v_seg,
                dropout_p=attn_dropout,
                softmax_scale=None,
                causal=True
            )  # [1,1,H,d]

            # ---- approx merge (length-weighted avg) ----
            w = float(ke - ks)
            merged_o = o * w if merged_o is None else (merged_o + o * w)
            wsum += w

        attn_output = (merged_o / max(wsum, 1e-6)).contiguous()  # [1,1,H,d]
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    # =========================
    # Prefill: q_len == kv_seq_len (FAST PATH: only block1)
    # =========================
    # Use block1 (standard RoPE) + single flash call, causal=True.
    qf = q1.transpose(1, 2).contiguous()                           # [1,L,H,d]
    kf = k1.transpose(1, 2).contiguous()                           # [1,L,H,d]
    vf = value_states_full.transpose(1, 2).contiguous()            # [1,L,H,d]

    attn_output = flash_attn_func(
        qf, kf, vf,
        dropout_p=attn_dropout,
        softmax_scale=None,
        causal=True
    )  # [1,L,H,d]

    attn_output = attn_output.contiguous().view(bsz, kv_seq_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value
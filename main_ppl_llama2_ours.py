# -*- coding: utf-8 -*-
"""
PPL evaluation with Llama-2 + Rodrope(flash_attn),
模型加载方式完全模仿 Needle-in-a-Haystack Rodrope 脚本。
"""

import os
import sys
import gc
import warnings
from typing import List

import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

warnings.filterwarnings("ignore")

# ====== Rodrope 相关 ======
import Rodrope
from torch import nn

# ==================== 配置：直接改这里 ====================
MODEL_PATH = "/home/liujia/llama-2-7b-chat"
# MODEL_PATH=/home/liujia/llama-2-7b-chat



# tokenized 好的 GovReport 测试集（你之前命令里的那个）
# TOKENIZED_DATASET_PATH = (
#     "/home/liujia/allcode/Extend_context_window_of_LLM/AEHVI-ROPE/datasets/emozilla-proofpile-test-tokenized-llama2/"
# )
TOKENIZED_DATASET_PATH = (
    "/home/liujia/allcode/Extend_context_window_of_LLM/AEHVI-ROPE/datasets/emozilla-govreport-test-tokenized-llama2/"
)

DATASET_MIN_TOKENS = 16384   # 过滤太短的样本
SAMPLES = 50                 # 只取前 50 个样本；设为 None 用全部

# PPL 相关参数（和你之前命令一致）
MIN_TOKENS = 2048
MAX_TOKENS = 16384
TOKENS_STEP = 1024
SLIDING_WINDOW = 512
TRUNCATE = True
AGGRESSIVE_MEMORY = True

# 输出 CSV 文件
OUTPUT_CSV = "/home/liujia/allcode/Extend_context_window_of_LLM/AEHVI-ROPE/result/PPL/llama2_govreport-RODROPE-window512-selected-configs.csv"

# Rodrope 参数：逐个测试这些候选配置的 PPL。
# block=2 时 m2/lambda3/m3/lambda4 写 None；block=3 时 m3/lambda4 写 None。
RODROPE_CONFIGS = [
    {"rodrope_block": 2, "m1": 512, "lambda2": 8.0, "m2": None, "lambda3": None, "m3": None, "lambda4": None},
    {"rodrope_block": 3, "m1": 512, "lambda2": 8.0, "m2": 10240, "lambda3": 8.0, "m3": None, "lambda4": None},
]
RODROPE_ENABLE_FLASH_ATTN = True
RODROPE_FLASH_IMPL = "flash_attn"   # 或 "triton"
RODROPE_SCALE_BASE = -1             # -1 表示不用 scale
# =====================================================


# ===== RoPE 适配：让 Rodrope 能调用 rope(seq_len=...) =====
class RodropeRopeAdapter(nn.Module):
    """
    包一层 LlamaRotaryEmbedding，使其兼容 Rodrope 内部
    调用形式 rope(seq_len=..., device=..., dtype=...).
    """

    def __init__(self, rope):
        super().__init__()
        self.rope = rope  # 原始 rotary_emb 模块

    def forward(self, *args, **kwargs):
        # Rodrope 调用时：rotary_emb(seq_len=..., device=..., dtype=...)
        if "seq_len" in kwargs and len(args) == 0:
            seq_len = kwargs.pop("seq_len")
            device = kwargs.pop("device", None)
            dtype = kwargs.pop("dtype", None)

            # transformers 4.38：通过 _set_cos_sin_cache 准备 cos/sin
            try:
                self.rope._set_cos_sin_cache(seq_len, device=device, dtype=dtype)
            except TypeError:
                # 某些版本签名为 _set_cos_sin_cache(self, seq_len, device=None, dtype=None)
                self.rope._set_cos_sin_cache(seq_len, device, dtype)
            return (self.rope.cos_cached, self.rope.sin_cached)

        # 正常 forward（模型原始调用路径）直接回退给原始模块
        return self.rope(*args, **kwargs)

    def __getattr__(self, name):
        # 透传其它属性（inv_freq, cos_cached 等）
        if name in {"rope", "forward", "__class__"}:
            return super().__getattr__(name)
        return getattr(self.rope, name)


def patch_rope_for_rodrope(model):
    """
    在所有层的 self_attn.rotary_emb 上包一层 RodropeRopeAdapter，
    使得 Rodrope.apply 内部的调用可以正常工作。
    """
    base = getattr(model, "model", None) or model
    n_wrapped = 0
    for layer in getattr(base, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "rotary_emb"):
            rope = attn.rotary_emb
            if not isinstance(rope, RodropeRopeAdapter):
                attn.rotary_emb = RodropeRopeAdapter(rope)
                n_wrapped += 1
    print(f"[patch_rope] wrapped rotary_emb in {n_wrapped} layers.")
    return n_wrapped > 0


def normalize_se_config(rodrope_config: dict) -> dict:
    rodrope_block = int(rodrope_config["rodrope_block"])
    if rodrope_block not in (2, 3, 4):
        raise ValueError(f"rodrope_block must be 2, 3 or 4, got {rodrope_block}")

    normalized = {
        "rodrope_block": rodrope_block,
        "m1": int(rodrope_config["m1"]),
        "lambda2": float(rodrope_config["lambda2"]),
    }
    if rodrope_block == 2:
        normalized.update({"m2": None, "lambda3": None, "m3": None, "lambda4": None})
        return normalized

    if rodrope_config.get("m2") is None or rodrope_config.get("lambda3") is None:
        raise ValueError("block>=3 requires m2 and lambda3.")
    if int(rodrope_config["m2"]) <= int(rodrope_config["m1"]):
        raise ValueError("For block>=3, m2 must be larger than m1.")
    normalized["m2"] = int(rodrope_config["m2"])
    normalized["lambda3"] = float(rodrope_config["lambda3"])

    if rodrope_block == 3:
        normalized.update({"m3": None, "lambda4": None})
        return normalized

    if rodrope_config.get("m3") is None or rodrope_config.get("lambda4") is None:
        raise ValueError("block=4 requires m3 and lambda4.")
    if int(rodrope_config["m3"]) <= int(rodrope_config["m2"]):
        raise ValueError("For block=4, m3 must be larger than m2.")
    normalized["m3"] = int(rodrope_config["m3"])
    normalized["lambda4"] = float(rodrope_config["lambda4"])
    return normalized


# ====== 模型加载：完全模仿 needle_in_haystack_Rodrope.py ======
def load_model_and_tokenizer(model_path: str, rodrope_config: dict):
    print(f"[load] loading from {model_path}")
    rodrope_config = normalize_se_config(rodrope_config)
    rodrope_block = rodrope_config["rodrope_block"]
    m1 = rodrope_config["m1"]
    lambda2 = rodrope_config["lambda2"]
    m2 = rodrope_config["m2"]
    lambda3 = rodrope_config["lambda3"]
    m3 = rodrope_config["m3"]
    lambda4 = rodrope_config["lambda4"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cfg = AutoConfig.from_pretrained(model_path)
    # 如果以后改成 Mistral，可以在这里关闭 sliding_window
    if getattr(cfg, "model_type", "") == "mistral":
        cfg.sliding_window = None

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        print("[load] success with attn_implementation=flash_attention_2")
    except Exception as e:
        print(f"[warn] flash_attention_2 load failed: {e}\n       fallback to eager.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=dtype,
            attn_implementation="eager",
            device_map="auto",
        )
        print("[load] success with attn_implementation=eager")

    # RoPE 适配 + Rodrope.apply（flash_attn 模式）
    patch_rope_for_rodrope(model)
    apply_kwargs = dict(
        group_size=lambda2,
        window_size=m1,
        enable_flash_attention=RODROPE_ENABLE_FLASH_ATTN,
        flash_attention_impl=RODROPE_FLASH_IMPL,
        scale_base=RODROPE_SCALE_BASE,
        block=rodrope_block,
    )
    if rodrope_block >= 3:
        apply_kwargs.update(
            far_size=m2,
            far_group_size=lambda3,
        )
    if rodrope_block == 4:
        apply_kwargs.update(
            far2_size=m3,
            far2_group_size=lambda4,
        )
    Rodrope.apply(model, **apply_kwargs)

    print(
        f"[Rodrope] applied: block={rodrope_block}, m1={m1}, lambda2={lambda2}, "
        f"m2={m2}, lambda3={lambda3}, m3={m3}, lambda4={lambda4}, "
        f"enable_flash_attention={RODROPE_ENABLE_FLASH_ATTN}, "
        f"flash_impl={RODROPE_FLASH_IMPL}, scale_base={RODROPE_SCALE_BASE}"
    )

    model.eval()
    return model, tokenizer


# ====== 数据集加载 ======
def load_tokenized_dataset(tokenized_path: str, dataset_min_tokens: int = None, samples: int = None):
    """
    既支持 save_to_disk 目录，也支持 Parquet 文件/目录。
    - 如果 tokenized_path 是一个包含 parquet 的目录，或直接是 .parquet 文件，
      走 datasets.load_dataset("parquet", data_files=...).
    - 否则尝试用 load_from_disk().
    """
    print(f"[data] loading tokenized dataset from {tokenized_path}")

    def _has_parquet(p):
        if os.path.isfile(p) and p.endswith(".parquet"):
            return True
        if os.path.isdir(p):
            for n in os.listdir(p):
                if n.endswith(".parquet"):
                    return True
        return False

    if _has_parquet(tokenized_path):
        # 目录里是 parquet 分片，或 path 本身就是一个 parquet 文件
        if os.path.isdir(tokenized_path):
            data_files = {"test": os.path.join(tokenized_path, "*.parquet")}
        else:
            data_files = {"test": tokenized_path}
        ds_dict = datasets.load_dataset("parquet", data_files=data_files)
        ds = ds_dict["test"]
        print(f"[data] loaded parquet split: {len(ds)}")
    else:
        # 尝试当作 save_to_disk 目录
        ds = datasets.load_from_disk(tokenized_path)
        print(f"[data] loaded saved-to-disk dataset: {len(ds)}")

    # 可能有的 tokenized parquet 只有原始 text，这里兜底：没有 input_ids/attention_mask 就自动分词
    cols = set(ds.column_names)
    if not ({"input_ids", "attention_mask"} <= cols):
        print("[data] 'input_ids'/'attention_mask' not found; tokenizing on-the-fly...")
        # 这里复用已加载的 tokenizer（在外部 load_model_and_tokenizer 已经创建）
        from transformers import AutoTokenizer
        # 保险起见，再取一遍 tokenizer（也可以把 tokenizer 作为参数传进来）
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def _tokenize(example):
            out = tok(
                example["text"],
                add_special_tokens=False,
                padding=False,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = out["input_ids"]
            example["attention_mask"] = out["attention_mask"]
            example["tokenized_len"] = len(out["input_ids"])
            return example

        ds = ds.map(_tokenize, desc="[data] tokenizing")

    # 过滤 & 采样
    if dataset_min_tokens:
        # 注意：有的 parquet 可能没有 tokenized_len，这里若缺失就现场计算一次
        if "tokenized_len" not in ds.column_names:
            ds = ds.map(lambda x: {"tokenized_len": len(x["input_ids"])}, desc="[data] add tokenized_len")
        ds = ds.filter(lambda x: x["tokenized_len"] >= dataset_min_tokens)
        print(f"[data] after filter tokenized_len >= {dataset_min_tokens}: {len(ds)} samples")

    if samples:
        ds = ds.select(range(min(samples, len(ds))))
        print(f"[data] take first {len(ds)} samples")

    return ds


def build_token_lengths(min_tokens: int, max_tokens: int, step: int) -> List[int]:
    return list(range(min_tokens, max_tokens + 1, step))


# ====== PPL 计算：只是 sliding-window forward，不改结构 ======
@torch.no_grad()
def compute_perplexity(
    encodings,
    model,
    tokenizer,
    add_start_token: bool = True,
    max_length: int = None,
    sliding_window: int = 256,
    truncate: bool = False,
    aggressive_memory: bool = False,
):
    """
    Sliding-window PPL，和你原始的 eval/ours.py 思路一致，只是去掉了 .to(device)，
    模型用 device_map="auto" + Rodrope(flash_attn)。
    """
    if add_start_token:
        assert tokenizer.bos_token is not None, (
            "模型需要 BOS token，如果没有，请把 add_start_token=False"
        )
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts))
    nlls = []

    for encoding_index in range(len(encoded_texts)):
        labels = torch.tensor(encoded_texts[encoding_index:encoding_index + 1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window):
            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc

            # 不手动 .to(device)，模仿 Needle 脚本（交给 HF/accelerate 处理）
            input_ids = labels[:, begin_loc:end_loc]

            if add_start_token:
                bos = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(0),
                    dtype=input_ids.dtype,
                )
                input_ids = torch.cat([bos, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

            if aggressive_memory:
                outputs = None
                input_ids = None
                target_ids = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood.detach().float().cpu())
            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}


# ====== 主函数：直接跑一个 py 文件 ======
def cleanup_model(model=None, tokenizer=None):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            with torch.cuda.device(dev_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 1. 加载 tokenized 数据集。数据与模型无关，只加载一次。
    ds = load_tokenized_dataset(
        TOKENIZED_DATASET_PATH,
        dataset_min_tokens=DATASET_MIN_TOKENS,
        samples=SAMPLES,
    )

    # 2. 构造不同的 max_length（token window）
    tokens_list = build_token_lengths(MIN_TOKENS, MAX_TOKENS, TOKENS_STEP)
    print(f"[ppl] token windows: {tokens_list}")
    print(f"[config] Rodrope configs: {RODROPE_CONFIGS}")

    # 3. 写出 CSV 表头，并逐个 Rodrope 配置追加结果，避免长任务中途丢数据。
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        header = [
            "model",
            "rodrope_block",
            "m1",
            "lambda2",
            "m2",
            "lambda3",
            "m3",
            "lambda4",
        ] + [str(x) for x in tokens_list]
        f.write(",".join(header) + "\n")

    for config_index, raw_se_config in enumerate(RODROPE_CONFIGS, start=1):
        model = None
        tokenizer = None
        try:
            rodrope_config = normalize_se_config(raw_se_config)
            print("=" * 80)
            print(f"[run] config_index={config_index}, config={rodrope_config}")

            # 每组 Rodrope 参数单独加载模型，避免 forward monkey-patch 参数互相污染。
            model, tokenizer = load_model_and_tokenizer(
                MODEL_PATH,
                rodrope_config=rodrope_config,
            )

            ppl_values = []
            for max_len in tokens_list:
                ppl = compute_perplexity(
                    encodings=ds,
                    model=model,
                    tokenizer=tokenizer,
                    add_start_token=(tokenizer.bos_token is not None),
                    max_length=max_len,
                    sliding_window=SLIDING_WINDOW,
                    truncate=TRUNCATE,
                    aggressive_memory=AGGRESSIVE_MEMORY,
                )["mean_perplexity"]

                print(
                    f"[ppl] {MODEL_PATH}: block={rodrope_config['rodrope_block']}, "
                    f"m1={rodrope_config['m1']}, lambda2={rodrope_config['lambda2']}, "
                    f"m2={rodrope_config['m2']}, lambda3={rodrope_config['lambda3']}, "
                    f"m3={rodrope_config['m3']}, lambda4={rodrope_config['lambda4']}, "
                    f"max_len={max_len} -> ppl={ppl}"
                )
                ppl_values.append(ppl)

            with open(OUTPUT_CSV, "a", encoding="utf-8") as f:
                row = [
                    MODEL_PATH,
                    str(rodrope_config["rodrope_block"]),
                    str(rodrope_config["m1"]),
                    str(rodrope_config["lambda2"]),
                    str(rodrope_config.get("m2")),
                    str(rodrope_config.get("lambda3")),
                    str(rodrope_config.get("m3")),
                    str(rodrope_config.get("lambda4")),
                ] + [str(x) for x in ppl_values]
                f.write(",".join(row) + "\n")

            print(f"[done] appended config_index={config_index} results to: {OUTPUT_CSV}")
        finally:
            cleanup_model(model, tokenizer)

    print(f"[done] all PPL results written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

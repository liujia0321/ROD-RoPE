# -*- coding: utf-8 -*-


import os
import sys
import gc
import warnings
import argparse
import csv
import json
from typing import Any, Dict, List, Optional

import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

warnings.filterwarnings("ignore")


import Rodrope
from torch import nn

# ==================== 配置：直接改这里 ====================
MODEL_PATH = "/path/to/Mistral-7B-Instruct-v0.1"

# tokenized 好的 Mistral GovReport 测试集，路径来自 main_ppl_mistral_self.py
# TOKENIZED_DATASET_PATH = (
#     "/path/to/emozillaproofpile-test-tokenized-mistral/"
# )

TOKENIZED_DATASET_PATH = (
    "/path/to/emozillagovreport-test-tokenized-mistral/"
)

DATASET_MIN_TOKENS = 16384   # 过滤太短的样本
SAMPLES = 50                 # 只取前 50 个样本；设为 None 用全部

# PPL 相关参数（和你之前命令一致）
MIN_TOKENS = 2048
MAX_TOKENS = 32768
TOKENS_STEP = 1024
SLIDING_WINDOW = 256
TRUNCATE = True
AGGRESSIVE_MEMORY = True


RODROPE_BLOCK = 2                  # 2/3/4-block Rodrope

# 输出 CSV 文件；命令行未指定 --output_csv 时会按 block 自动生成
OUTPUT_CSV = None
M1 = 2048                      # neighbor window size
LAMBDA2 = 9.0                  # compression ratio for the first grouped block
M2 = 12288                     # distance boundary where the far block starts
LAMBDA3 = 32.0                 # compression ratio for the far block
M3 = None                      # distance boundary where the far2 block starts
LAMBDA4 = None                 # compression ratio for the far2 block
RODROPE_ENABLE_FLASH_ATTN = True
RODROPE_FLASH_IMPL = "flash_attn"   # fixed-block path currently supports "flash_attn"
RODROPE_SCALE_BASE = -1             # -1 表示不用 scale

# 批量评估配置：直接在这里填任意 Rodrope 配置即可。
# 设为 [] 或 None 时，会退回到上面的单配置 RODROPE_BLOCK/M1/LAMBDA2/...。
RODROPE_CONFIGS = [
    {"rodrope_block": 2, "m1": 2048, "lambda2": 9.0, "m2": None, "lambda3": None, "m3": None, "lambda4": None},
    {"rodrope_block": 3, "m1": 128, "lambda2": 8.0, "m2": 12288, "lambda3": 32.0, "m3": None, "lambda4": None},
]
# =====================================================



class RodropeRopeAdapter(nn.Module):
    """
    包一层 LlamaRotaryEmbedding，使其兼容 Rodrope 内部
    调用形式 rope(seq_len=..., device=..., dtype=...).
    """

    def __init__(self, rope):
        super().__init__()
        self.rope = rope  # 原始 rotary_emb 模块

    def forward(self, *args, **kwargs):

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



def load_model_and_tokenizer(
    model_path: str,
    rodrope_block: int,
    m1: int,
    lambda2: float,
    m2: int = None,
    lambda3: float = None,
    m3: int = None,
    lambda4: float = None,
    rodrope_enable_flash_attention: bool = True,
    rodrope_flash_impl: str = "flash_attn",
    rodrope_scale_base: int = -1,
):
    print(f"[load] loading from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cfg = AutoConfig.from_pretrained(model_path)
    # Mistral 需要关闭 sliding_window，否则长上下文和 Rodrope 容易冲突
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
        enable_flash_attention=rodrope_enable_flash_attention,
        flash_attention_impl=rodrope_flash_impl,
        scale_base=rodrope_scale_base,
        block=rodrope_block,
    )
    if rodrope_block >= 3:
        if m2 is None or lambda3 is None:
            raise ValueError("block>=3 requires m2 and lambda3.")
        apply_kwargs.update(
            far_size=m2,
            far_group_size=lambda3,
        )
    if rodrope_block == 4:
        if m3 is None or lambda4 is None:
            raise ValueError("block=4 requires m3 and lambda4.")
        apply_kwargs.update(
            far2_size=m3,
            far2_group_size=lambda4,
        )
    Rodrope.apply(model, **apply_kwargs)

    print(
        f"[Rodrope] applied: block={rodrope_block}, m1={m1}, lambda2={lambda2}, "
        f"m2={m2}, lambda3={lambda3}, m3={m3}, lambda4={lambda4}, "
        f"enable_flash_attention={rodrope_enable_flash_attention}, "
        f"flash_impl={rodrope_flash_impl}, scale_base={rodrope_scale_base}"
    )

    model.eval()
    return model, tokenizer


# ====== 数据集加载 ======
def load_tokenized_dataset(tokenized_path: str, dataset_min_tokens: int = None, samples: int = None, tokenizer_path: str = MODEL_PATH):
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
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
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

            nlls.append(neg_log_likelihood)
            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}


def parse_args():
    parser = argparse.ArgumentParser(description="PPL evaluation with Mistral + configurable Rodrope")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="path to model")
    parser.add_argument("--tokenized_dataset_path", type=str, default=TOKENIZED_DATASET_PATH, help="tokenized dataset path")
    parser.add_argument("--dataset_min_tokens", type=int, default=DATASET_MIN_TOKENS, help="filter samples shorter than this")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="number of samples; <=0 means all")
    parser.add_argument("--min_tokens", type=int, default=MIN_TOKENS, help="minimum evaluated max_length")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="maximum evaluated max_length")
    parser.add_argument("--tokens_step", type=int, default=TOKENS_STEP, help="step between evaluated max_length values")
    parser.add_argument("--sliding_window", type=int, default=SLIDING_WINDOW, help="sliding window stride")
    parser.add_argument("--no_truncate", action="store_true", help="disable truncation to each max_length")
    parser.add_argument("--no_aggressive_memory", action="store_true", help="disable aggressive CUDA memory cleanup")
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV, help="output CSV path; default is generated from settings")
    parser.add_argument(
        "--rodrope_configs",
        type=str,
        default=None,
        help="JSON string or JSON file path containing a list of Rodrope configs; overrides RODROPE_CONFIGS",
    )

    parser.add_argument("--rodrope_block", type=int, default=RODROPE_BLOCK, choices=[2, 3, 4], help="2/3/4-block Rodrope")
    parser.add_argument("--m1", type=int, default=M1, help="neighbor window size")
    parser.add_argument("--lambda2", type=float, default=LAMBDA2, help="compression ratio for the first grouped block")
    parser.add_argument("--m2", type=int, default=M2, help="distance boundary where the far block starts")
    parser.add_argument("--lambda3", type=float, default=LAMBDA3, help="compression ratio for the far block")
    parser.add_argument("--m3", type=int, default=M3, help="distance boundary where the far2 block starts")
    parser.add_argument("--lambda4", type=float, default=LAMBDA4, help="compression ratio for the far2 block")
    parser.add_argument("--rodrope_flash_impl", type=str, default=RODROPE_FLASH_IMPL, choices=["flash_attn", "triton"], help="fixed-block path supports flash_attn")
    parser.add_argument("--rodrope_scale_base", type=int, default=RODROPE_SCALE_BASE, help="query scaling base; -1 disables scaling")
    parser.add_argument("--disable_flash_attention", action="store_true", help="use non-flash path; only valid for block=2")
    return parser.parse_args()


def _read_rodrope_configs_arg(configs_arg: Optional[str]):
    if not configs_arg:
        return None
    if os.path.exists(configs_arg):
        with open(configs_arg, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(configs_arg)


def _single_config_from_args(args) -> Dict[str, Any]:
    return {
        "rodrope_block": args.rodrope_block,
        "m1": args.m1,
        "lambda2": args.lambda2,
        "m2": args.m2,
        "lambda3": args.lambda3,
        "m3": args.m3,
        "lambda4": args.lambda4,
    }


def _normalize_rodrope_config(raw_config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(defaults)
    config.update(raw_config)

    rodrope_block = int(config["rodrope_block"])
    if rodrope_block not in (2, 3, 4):
        raise ValueError(f"rodrope_block must be 2, 3 or 4, got {rodrope_block}")

    normalized = {
        "rodrope_block": rodrope_block,
        "m1": int(config["m1"]),
        "lambda2": float(config["lambda2"]),
    }

    if rodrope_block == 2:
        normalized["m2"] = None
        normalized["lambda3"] = None
        normalized["m3"] = None
        normalized["lambda4"] = None
        return normalized

    if config.get("m2") is None or config.get("lambda3") is None:
        raise ValueError("block>=3 requires m2 and lambda3.")
    if int(config["m2"]) <= int(config["m1"]):
        raise ValueError("For block>=3, m2 must be larger than m1.")
    normalized["m2"] = int(config["m2"])
    normalized["lambda3"] = float(config["lambda3"])

    if rodrope_block == 3:
        normalized["m3"] = None
        normalized["lambda4"] = None
        return normalized

    if config.get("m3") is None or config.get("lambda4") is None:
        raise ValueError("block=4 requires m3 and lambda4.")
    if int(config["m3"]) <= int(config["m2"]):
        raise ValueError("For block=4, m3 must be larger than m2.")
    normalized["m3"] = int(config["m3"])
    normalized["lambda4"] = float(config["lambda4"])
    return normalized


def resolve_rodrope_configs(args):
    raw_configs = _read_rodrope_configs_arg(args.rodrope_configs)
    using_config_list = raw_configs is not None

    if raw_configs is None and RODROPE_CONFIGS:
        raw_configs = RODROPE_CONFIGS
        using_config_list = True

    if raw_configs is None:
        raw_configs = [_single_config_from_args(args)]

    if isinstance(raw_configs, dict):
        raw_configs = [raw_configs]
    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("RODROPE_CONFIGS must be a non-empty list of config dictionaries.")

    defaults = _single_config_from_args(args)
    configs = [_normalize_rodrope_config(config, defaults) for config in raw_configs]
    return configs, using_config_list


def build_output_csv(args, config=None, using_config_list=False):
    if args.output_csv:
        return args.output_csv
    if using_config_list:
        return "data/mistral_govreport-rodrope_configs.csv"

    config = config or _single_config_from_args(args)
    if config["rodrope_block"] == 2:
        suffix = f"2block_m1{config['m1']}_lambda2{config['lambda2']}"
    elif config["rodrope_block"] == 3:
        suffix = (
            f"3block_m1{config['m1']}_lambda2{config['lambda2']}"
            f"_m2{config['m2']}_lambda3{config['lambda3']}"
        )
    else:
        suffix = (
            f"4block_m1{config['m1']}_lambda2{config['lambda2']}"
            f"_m2{config['m2']}_lambda3{config['lambda3']}"
            f"_m3{config['m3']}_lambda4{config['lambda4']}"
        )
    return f"data/mistral_govreport-{suffix}.csv"


def _cleanup_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ====== 主函数：直接跑一个 py 文件 ======
def main():
    args = parse_args()
    rodrope_enable_flash_attention = not args.disable_flash_attention
    rodrope_configs, using_config_list = resolve_rodrope_configs(args)

    for config in rodrope_configs:
        if config["rodrope_block"] >= 3 and not rodrope_enable_flash_attention:
            raise NotImplementedError("block>=3 Rodrope is implemented for flash attention only.")
        if config["rodrope_block"] >= 3 and args.rodrope_flash_impl == "triton":
            raise NotImplementedError("block>=3 Rodrope is implemented for --rodrope_flash_impl flash_attn.")

    output_csv = build_output_csv(args, config=rodrope_configs[0], using_config_list=using_config_list)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 tokenized 数据集。多配置评估时，数据只需要加载一次。
    samples = None if args.samples is not None and args.samples <= 0 else args.samples
    ds = load_tokenized_dataset(
        args.tokenized_dataset_path,
        dataset_min_tokens=args.dataset_min_tokens,
        samples=samples,
        tokenizer_path=args.model_path,
    )

    # 2. 构造不同的 max_length（token window）
    tokens_list = build_token_lengths(args.min_tokens, args.max_tokens, args.tokens_step)
    print(f"[ppl] token windows: {tokens_list}")
    print(f"[config] evaluating {len(rodrope_configs)} Rodrope config(s)")

    rows = []
    for config_index, config in enumerate(rodrope_configs, start=1):
        print(
            f"\n[config {config_index}/{len(rodrope_configs)}] "
            f"block={config['rodrope_block']}, m1={config['m1']}, "
            f"lambda2={config['lambda2']}, m2={config['m2']}, "
            f"lambda3={config['lambda3']}, m3={config['m3']}, "
            f"lambda4={config['lambda4']}"
        )

        model, tokenizer = load_model_and_tokenizer(
            args.model_path,
            rodrope_block=config["rodrope_block"],
            m1=config["m1"],
            lambda2=config["lambda2"],
            m2=config["m2"],
            lambda3=config["lambda3"],
            m3=config["m3"],
            lambda4=config["lambda4"],
            rodrope_enable_flash_attention=rodrope_enable_flash_attention,
            rodrope_flash_impl=args.rodrope_flash_impl,
            rodrope_scale_base=args.rodrope_scale_base,
        )

        ppl_values = []
        for max_len in tokens_list:
            ppl = compute_perplexity(
                encodings=ds,
                model=model,
                tokenizer=tokenizer,
                add_start_token=(tokenizer.bos_token is not None),
                max_length=max_len,
                sliding_window=args.sliding_window,
                truncate=not args.no_truncate,
                aggressive_memory=not args.no_aggressive_memory,
            )["mean_perplexity"]

            print(
                f"[ppl] config={config_index}, model={args.model_path}: "
                f"max_len={max_len} -> ppl={ppl}"
            )
            ppl_values.append(ppl)

        if using_config_list:
            rows.append([
                args.model_path,
                config["rodrope_block"],
                config["m1"],
                config["lambda2"],
                config["m2"],
                config["lambda3"],
                config["m3"],
                config["lambda4"],
                *ppl_values,
            ])
        else:
            rows.append([args.model_path, *ppl_values])

        _cleanup_model(model, tokenizer)

    # 3. 写出 CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if using_config_list:
            writer.writerow([
                "model_path",
                "rodrope_block",
                "m1",
                "lambda2",
                "m2",
                "lambda3",
                "m3",
                "lambda4",
                *tokens_list,
            ])
        else:
            writer.writerow(["", *tokens_list])
        writer.writerows(rows)

    print(f"[done] PPL results written to: {output_csv}")


if __name__ == "__main__":
    main()

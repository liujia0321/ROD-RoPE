# -*- coding: utf-8 -*-
"""
LongBench evaluation for Llama-3-Instruct with ROD-RoPE.

Example:
CUDA_VISIBLE_DEVICES=0 python main_longbench_llama3_ours.py \
    --datasets narrativeqa qasper gov_report \
    --model_path /home/liujia/allcode/meta-llamaMeta-Llama-3-8B-Instruct \
    --max_length 32768 \
    --rodrope_block 3 --m1 256 --lambda2 6.85 --m2 4096 --lambda3 16.0 \
    --evaluate
"""

import argparse
import gc
import json
import os
import random
import sys
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import Rodrope

warnings.filterwarnings("ignore")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LONG_BENCH_ROOT = "/home/liujia/allcode/LongBench-main"
MODEL_PATH = "/home/liujia/allcode/meta-llamaMeta-Llama-3-8B-Instruct"

LONG_BENCH_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]

LONG_BENCH_E_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

RAW_PROMPT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


class RodropeRopeAdapter(nn.Module):
    """
    Wrap LlamaRotaryEmbedding so Rodrope internals can call
    rope(seq_len=..., device=..., dtype=...).
    """

    def __init__(self, rope):
        super().__init__()
        self.rope = rope

    def forward(self, *args, **kwargs):
        if "seq_len" in kwargs and len(args) == 0:
            seq_len = kwargs.pop("seq_len")
            device = kwargs.pop("device", None)
            dtype = kwargs.pop("dtype", None)
            try:
                self.rope._set_cos_sin_cache(seq_len, device=device, dtype=dtype)
            except TypeError:
                self.rope._set_cos_sin_cache(seq_len, device, dtype)
            return self.rope.cos_cached, self.rope.sin_cached
        return self.rope(*args, **kwargs)

    def __getattr__(self, name):
        if name in {"rope", "forward", "__class__"}:
            return super().__getattr__(name)
        return getattr(self.rope, name)


def patch_rope_for_rodrope(model) -> bool:
    base = getattr(model, "model", None) or model
    n_wrapped = 0
    for layer in getattr(base, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "rotary_emb"):
            continue
        rope = attn.rotary_emb
        if not isinstance(rope, RodropeRopeAdapter):
            attn.rotary_emb = RodropeRopeAdapter(rope)
            n_wrapped += 1
    print(f"[patch_rope] wrapped rotary_emb in {n_wrapped} layers.")
    return n_wrapped > 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_input_device(model) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_chat_prompt(tokenizer, prompt: str, dataset: str, use_chat_template: bool) -> str:
    if dataset in RAW_PROMPT_DATASETS or not use_chat_template:
        return prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def post_process(response: str) -> str:
    stop_tokens = [
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|im_end|>",
    ]
    for stop in stop_tokens:
        if stop in response:
            response = response.split(stop)[0]
    return response.strip()


def truncate_prompt_middle(tokenizer, prompt: str, max_length: int) -> str:
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) <= max_length:
        return prompt
    half = max_length // 2
    left = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
    right = tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    return left + right


def normalize_rodrope_config(args) -> Dict[str, Any]:
    block = int(args.rodrope_block)
    if block not in (2, 3, 4):
        raise ValueError("rodrope_block must be 2, 3 or 4.")

    config = {
        "rodrope_block": block,
        "m1": int(args.m1),
        "lambda2": float(args.lambda2),
        "m2": None,
        "lambda3": None,
        "m3": None,
        "lambda4": None,
    }
    if block >= 3:
        if args.m2 is None or args.lambda3 is None:
            raise ValueError("rodrope_block >= 3 requires --m2 and --lambda3.")
        if int(args.m2) <= int(args.m1):
            raise ValueError("--m2 must be larger than --m1.")
        config["m2"] = int(args.m2)
        config["lambda3"] = float(args.lambda3)
    if block == 4:
        if args.m3 is None or args.lambda4 is None:
            raise ValueError("rodrope_block == 4 requires --m3 and --lambda4.")
        if int(args.m3) <= int(args.m2):
            raise ValueError("--m3 must be larger than --m2.")
        config["m3"] = int(args.m3)
        config["lambda4"] = float(args.lambda4)
    return config


def build_run_name(model_path: str, config: Dict[str, Any], explicit_name: Optional[str]) -> str:
    if explicit_name:
        return explicit_name
    model_name = os.path.basename(os.path.normpath(model_path))
    if config["rodrope_block"] == 2:
        suffix = f"rodrope_2block_m1{config['m1']}_lambda2{config['lambda2']}"
    elif config["rodrope_block"] == 3:
        suffix = (
            f"rodrope_3block_m1{config['m1']}_lambda2{config['lambda2']}"
            f"_m2{config['m2']}_lambda3{config['lambda3']}"
        )
    else:
        suffix = (
            f"rodrope_4block_m1{config['m1']}_lambda2{config['lambda2']}"
            f"_m2{config['m2']}_lambda3{config['lambda3']}"
            f"_m3{config['m3']}_lambda4{config['lambda4']}"
        )
    return f"{model_name}_{suffix}".replace("/", "_")


def load_model_and_tokenizer(args, config: Dict[str, Any]):
    print(f"[load] loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if getattr(hf_config, "model_type", "") == "mistral":
        hf_config.sliding_window = None
    if args.max_position_embeddings:
        hf_config.max_position_embeddings = args.max_position_embeddings
    if args.no_use_cache:
        hf_config.use_cache = False

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = dict(
        config=hf_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device_map,
    )

    loaded_with_flash = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            **load_kwargs,
        )
        loaded_with_flash = True
        print("[load] success with attn_implementation=flash_attention_2")
    except Exception as exc:
        print(f"[warn] flash_attention_2 load failed: {exc}")
        if config["rodrope_block"] >= 3 and not args.disable_flash_attention:
            raise RuntimeError("ROD-RoPE block>=3 requires flash_attention_2 in this codebase.") from exc
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="eager",
            **load_kwargs,
        )
        print("[load] fallback success with attn_implementation=eager")

    patch_rope_for_rodrope(model)
    rodrope_enable_flash_attention = (not args.disable_flash_attention) and loaded_with_flash
    Rodrope.apply(
        model,
        group_size=config["lambda2"],
        window_size=config["m1"],
        far_size=config["m2"],
        far_group_size=config["lambda3"],
        far2_size=config["m3"],
        far2_group_size=config["lambda4"],
        block=config["rodrope_block"],
        enable_flash_attention=rodrope_enable_flash_attention,
        flash_attention_impl=args.rodrope_flash_impl,
        scale_base=args.rodrope_scale_base,
    )
    print(
        f"[Rodrope] block={config['rodrope_block']}, m1={config['m1']}, "
        f"lambda2={config['lambda2']}, m2={config['m2']}, lambda3={config['lambda3']}, "
        f"m3={config['m3']}, lambda4={config['lambda4']}, "
        f"enable_flash_attention={rodrope_enable_flash_attention}, "
        f"flash_impl={args.rodrope_flash_impl}, scale_base={args.rodrope_scale_base}"
    )
    model.eval()
    return model, tokenizer


def resolve_datasets(args) -> List[str]:
    if args.datasets:
        return args.datasets
    return LONG_BENCH_E_DATASETS if args.e else LONG_BENCH_DATASETS


def load_longbench_dataset(args, dataset: str) -> List[Dict[str, Any]]:
    if args.use_hf:
        hf_name = f"{dataset}_e" if args.e else dataset
        return list(load_dataset("THUDM/LongBench", hf_name, split="test"))

    filename = f"{dataset}_e.jsonl" if args.e else f"{dataset}.jsonl"
    path = os.path.join(args.data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LongBench data file not found: {path}")
    return load_jsonl(path)


def read_done_count(out_path: str) -> int:
    if not os.path.exists(out_path):
        return 0
    with open(out_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    dataset: str,
    max_gen: int,
    device: torch.device,
) -> str:
    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = inputs.input_ids.shape[-1]
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    if dataset == "samsum":
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        generation_kwargs["min_length"] = context_length + 1
        generation_kwargs["eos_token_id"] = [tokenizer.eos_token_id, newline_id]

    output = model.generate(**generation_kwargs)[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return post_process(pred)


def predict_dataset(
    args,
    model,
    tokenizer,
    dataset: str,
    data: Sequence[Dict[str, Any]],
    prompt_format: str,
    max_gen: int,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done_count = read_done_count(out_path) if args.resume else 0
    if done_count and done_count >= len(data):
        print(f"[skip] {dataset}: {done_count}/{len(data)} samples already exist.")
        return
    if not args.resume and os.path.exists(out_path):
        open(out_path, "w", encoding="utf-8").close()

    device = get_input_device(model)
    selected = data[done_count:]
    if args.samples is not None and args.samples > 0:
        selected = selected[: max(args.samples - done_count, 0)]
    print(f"[predict] {dataset}: writing to {out_path}; start at sample {done_count}")

    with open(out_path, "a", encoding="utf-8") as f:
        for json_obj in tqdm(selected, desc=dataset):
            prompt = prompt_format.format(**json_obj)
            prompt = truncate_prompt_middle(tokenizer, prompt, args.max_length)
            prompt = build_chat_prompt(tokenizer, prompt, dataset, args.use_chat_template)
            pred = generate_one(model, tokenizer, prompt, dataset, max_gen, device)
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj.get("all_classes", []),
                    "length": json_obj.get("length", None),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
            f.flush()

            if args.aggressive_memory:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def import_longbench_metrics(longbench_root: str):
    if longbench_root not in sys.path:
        sys.path.insert(0, longbench_root)
    from metrics import (  # pylint: disable=import-error,import-outside-toplevel
        classification_score,
        code_sim_score,
        count_score,
        qa_f1_score,
        qa_f1_zh_score,
        retrieval_score,
        retrieval_zh_score,
        rouge_score,
        rouge_zh_score,
    )

    return {
        "narrativeqa": qa_f1_score,
        "qasper": qa_f1_score,
        "multifieldqa_en": qa_f1_score,
        "multifieldqa_zh": qa_f1_zh_score,
        "hotpotqa": qa_f1_score,
        "2wikimqa": qa_f1_score,
        "musique": qa_f1_score,
        "dureader": rouge_zh_score,
        "gov_report": rouge_score,
        "qmsum": rouge_score,
        "multi_news": rouge_score,
        "vcsum": rouge_zh_score,
        "trec": classification_score,
        "triviaqa": qa_f1_score,
        "samsum": rouge_score,
        "lsht": classification_score,
        "passage_retrieval_en": retrieval_score,
        "passage_count": count_score,
        "passage_retrieval_zh": retrieval_zh_score,
        "lcc": code_sim_score,
        "repobench-p": code_sim_score,
    }


def score_longbench_dataset(
    dataset: str,
    predictions: Sequence[str],
    answers: Sequence[Sequence[str]],
    all_classes,
    metrics: Dict[str, Any],
) -> float:
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, metrics[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2) if predictions else 0.0


def score_longbench_e_dataset(
    dataset: str,
    predictions: Sequence[str],
    answers: Sequence[Sequence[str]],
    lengths: Sequence[Optional[int]],
    all_classes,
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in {"trec", "triviaqa", "samsum", "lsht"}:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, metrics[dataset](prediction, ground_truth, all_classes=all_classes))
        if length is not None and length < 4000:
            scores["0-4k"].append(score)
        elif length is not None and length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    return {
        key: round(100 * float(np.mean(value)), 2) if value else 0.0
        for key, value in scores.items()
    }


def evaluate_predictions(args, pred_dir: str, datasets: Iterable[str]) -> Dict[str, Any]:
    metrics = import_longbench_metrics(args.longbench_root)
    scores = {}
    for dataset in datasets:
        path = os.path.join(pred_dir, f"{dataset}.jsonl")
        if not os.path.exists(path):
            print(f"[eval-skip] missing predictions: {path}")
            continue
        predictions, answers, lengths = [], [], []
        all_classes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                predictions.append(row["pred"])
                answers.append(row["answers"])
                all_classes = row.get("all_classes", [])
                lengths.append(row.get("length", None))
        if args.e:
            scores[dataset] = score_longbench_e_dataset(
                dataset, predictions, answers, lengths, all_classes, metrics
            )
        else:
            scores[dataset] = score_longbench_dataset(
                dataset, predictions, answers, all_classes, metrics
            )

    result_path = os.path.join(pred_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(f"[eval] results written to {result_path}")
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="LongBench with Llama-3-Instruct + ROD-RoPE")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--longbench_root", type=str, default=LONG_BENCH_ROOT)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=os.path.join(SCRIPT_DIR, "pred_longbench"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--use_hf", action="store_true", help="Load THUDM/LongBench from Hugging Face")
    parser.add_argument("--samples", type=int, default=None, help="Limit samples per dataset for debugging")
    parser.add_argument("--resume", action="store_true", help="Append missing samples to existing jsonl files")
    parser.add_argument("--evaluate", action="store_true", help="Score predictions after generation")
    parser.add_argument("--eval_only", action="store_true", help="Only score existing predictions")

    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--no_use_cache", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--aggressive_memory", action="store_true")

    parser.add_argument("--rodrope_block", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--m1", type=int, default=256)
    parser.add_argument("--lambda2", type=float, default=6.85)
    parser.add_argument("--m2", type=int, default=4096)
    parser.add_argument("--lambda3", type=float, default=16.0)
    parser.add_argument("--m3", type=int, default=None)
    parser.add_argument("--lambda4", type=float, default=None)
    parser.add_argument("--rodrope_flash_impl", type=str, default="flash_attn", choices=["flash_attn", "triton"])
    parser.add_argument("--rodrope_scale_base", type=int, default=-1)
    parser.add_argument("--disable_flash_attention", action="store_true")
    args = parser.parse_args()

    args.data_dir = args.data_dir or os.path.join(args.longbench_root, "longbench", "data")
    args.config_dir = args.config_dir or os.path.join(args.longbench_root, "config")
    args.use_chat_template = not args.no_chat_template
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    config = normalize_rodrope_config(args)
    run_name = build_run_name(args.model_path, config, args.run_name)
    split_root = "pred_e" if args.e else "pred"
    pred_dir = os.path.join(args.output_root, split_root, run_name)
    os.makedirs(pred_dir, exist_ok=True)

    datasets = resolve_datasets(args)
    dataset2prompt = load_json(os.path.join(args.config_dir, "dataset2prompt.json"))
    dataset2maxlen = load_json(os.path.join(args.config_dir, "dataset2maxlen.json"))

    print(f"[config] run_name={run_name}")
    print(f"[config] pred_dir={pred_dir}")
    print(f"[config] datasets={datasets}")

    if not args.eval_only:
        model, tokenizer = load_model_and_tokenizer(args, config)
        for dataset in datasets:
            data = load_longbench_dataset(args, dataset)
            if args.samples is not None and args.samples > 0 and not args.resume:
                data = data[: args.samples]
            predict_dataset(
                args=args,
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                data=data,
                prompt_format=dataset2prompt[dataset],
                max_gen=int(dataset2maxlen[dataset]),
                out_path=os.path.join(pred_dir, f"{dataset}.jsonl"),
            )

    if args.evaluate or args.eval_only:
        evaluate_predictions(args, pred_dir, datasets)

    print(f"[done] predictions are under: {pred_dir}")


if __name__ == "__main__":
    main()

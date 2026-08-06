# -*- coding: utf-8 -*-
"""
Run Counting-Stars acquisition/reasoning only for Llama-2 + Rodrope.

This is a standalone counting evaluator extracted from optimizatin_allblock_llama2.py.
It does not load the PPL dataset and does not compute PPL.
"""

import argparse
import gc
import glob
import json
import os
import re
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import json_repair
import Rodrope

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==================== default config: keep aligned with optimizatin_allblock_llama2.py ====================
MODEL_PATH = "/path/to/llama-2-7b-chat"
RESULTS_ROOT = "/path/to/result/counting-stars/llama2"
COUNTING_CONTEXT_ROOT = "/path/to/context_data"
ACQ_PDF_PATH = "/path/to/result/counting-stars/llama2/acq.pdf"
REA_PDF_PATH = "/path/to/result/counting-stars/llama2/rea.pdf"

LANGUAGE = "EN"
M = 8
N = 8
MAX_CONTEXT_LENGTH = 16384
RUN_TAG = "manual_counting_llama2"

RODROPE_BLOCK = 3
M1 = 128
LAMBDA2 = 8.0
M2 = 12288
LAMBDA3 = 32.0
M3 = None
LAMBDA4 = None
RODROPE_FLASH_IMPL = "flash_attn"

# 直接在这里写要测试的 Rodrope 配置；脚本会依次测试每一组配置的 acquisition/reasoning。
# block=2 时 m2/lambda3/m3/lambda4 写 None 即可；block=3 时 m3/lambda4 写 None 即可。
COUNTING_CONFIGS = [
    {"rodrope_block": 2, "m1": 256, "lambda2": 9.54, "m2": None, "lambda3": None, "m3": None, "lambda4": None},
]
# ==========================================================================================================

class RodropeRopeAdapter(nn.Module):
    """Make LlamaRotaryEmbedding compatible with Rodrope rope(seq_len=..., device=..., dtype=...)."""

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
        if attn is not None and hasattr(attn, "rotary_emb"):
            rope = attn.rotary_emb
            if not isinstance(rope, RodropeRopeAdapter):
                attn.rotary_emb = RodropeRopeAdapter(rope)
                n_wrapped += 1
    print(f"[patch_rope] wrapped rotary_emb in {n_wrapped} layers.", flush=True)
    return n_wrapped > 0


def format_float_for_filename(value: float, digits: int = 6) -> str:
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def get_context(language: str) -> str:
    if language == "EN":
        context = ""
        for file in glob.glob(os.path.join(COUNTING_CONTEXT_ROOT, "PaulGrahamEssays", "*.txt")):
            with open(file, "r", encoding="utf-8") as f:
                context += f.read().replace("\n", " ")
        return context
    if language == "ZH":
        string_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        with open(os.path.join(COUNTING_CONTEXT_ROOT, "The_Story_of_the_Stone.txt"), "r", encoding="utf-8") as f:
            context = ""
            for line in f.readlines():
                context += line.strip().replace("------------", " ").replace("\n", " ").replace(" ", "")
        context = re.sub("[{}]".format(string_punctuation), "", context)
        context = re.sub("[a-zA-Z]", "", context)
        return context
    raise ValueError(f"Unknown language: {language}")


def get_stars(stars_path: str, counting_times: int) -> List[int]:
    with open(stars_path, "r", encoding="utf-8") as stars_file:
        return eval(stars_file.readline())[str(counting_times)]


def sentence_with_star(language: str, test_type: str, indicator: int, a_stars: List[int], r_stars: List[int]) -> str:
    tt = test_type.lower()
    if language == "ZH":
        if tt == "acquisition":
            return f"\n小企鹅数了{a_stars[indicator]}颗★\n"
        if tt == "reasoning":
            return f"\n小企鹅数了{r_stars[indicator]}颗★，但发现数错了，于是又数了一遍，这次数对了，是{a_stars[indicator]}颗★\n"
    elif language == "EN":
        if tt == "acquisition":
            return f"\nThe little penguin counted {a_stars[indicator]} ★\n"
        if tt == "reasoning":
            return (
                f"\nThe little penguin counted {r_stars[indicator]} ★, "
                f"but found that a mistake had been made, so the counting was done again, "
                f"and this time {a_stars[indicator]} ★ was counted correctly.\n"
            )
    raise ValueError(f"Unknown language/test_type: {language}/{test_type}")


def select_question(language: str, test_type: str) -> str:
    tt = test_type.lower()
    if language == "ZH":
        if tt == "acquisition":
            return (
                "\n\n\n\n在这个月光皎洁、云雾缭绕的夜晚，小企鹅正望向天空，全神贯注地数★。"
                "请帮助小企鹅收集所数★的颗数，按照如下格式：{\"小企鹅\":[x,x,x,...]}，不要求和，"
                "[x,x,x,...]中数字为小企鹅每次数★的颗数，仅以JSON格式输出结果，不需要输出任何解释。"
            )
        if tt == "reasoning":
            return (
                "\n\n\n\n在这个月光皎洁、云雾缭绕的夜晚，小企鹅正望向天空，全神贯注地数★。"
                "请帮助小企鹅收集所数★的正确颗数，按照如下格式：{\"小企鹅\":[x,x,x,...]}，不要求和，"
                "[x,x,x,...]中数字为小企鹅正确数★的颗数，仅以JSON格式输出结果，不需要输出任何解释。"
            )
    elif language == "EN":
        if tt == "acquisition":
            return (
                "\n\n\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. "
                "Please help the little penguin collect the number of ★, for example: {\"little_penguin\": [x, x, x,...]}. "
                "The summation is not required, and the numbers in [x, x, x,...] represent the counted number of ★ by the little penguin. "
                "Only output the results in JSON format without any explanation."
            )
        if tt == "reasoning":
            return (
                "\n\n\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. "
                "Please help the little penguin collect the correct number of ★, for example: {\"little_penguin\": [x, x, x,...]}. "
                "The summation is not required, and the numbers in [x, x, x,...] represent the correctly counted number of ★ by the little penguin. "
                "Only output the results in JSON format without any explanation."
            )
    raise ValueError(f"Unknown language/test_type: {language}/{test_type}")


def generate_prompt(context: str, retrieval_question: str) -> str:
    return (
        f" This is a very long story book: <book> {context} </book>.\n"
        f" Based on the content of the book, Question: {retrieval_question}\nAnswer:"
    )


def extract_numbers_from_string(string: str) -> List[int]:
    numbers = re.findall(r"\d+", str(string))
    return [int(num) for num in numbers] if numbers else []


def reduce_duplicate(predicted: List[int], m: int) -> List[int]:
    if len(predicted) > m:
        return list(set(predicted[:m]))
    return list(set(predicted))


def get_reasoning_score(index: int, predicted: List[int], a_stars: List[int], r_stars: List[int]) -> float:
    if a_stars[index] in predicted and r_stars[index] in predicted:
        return 0.5
    if a_stars[index] in predicted and r_stars[index] not in predicted:
        return 1.0
    if a_stars[index] not in predicted and r_stars[index] in predicted:
        return 0.25
    return 0.0


def iter_jsonl(file_obj) -> Iterable[dict]:
    for line in file_obj:
        line = line.strip()
        if line:
            yield json.loads(line)


def append_jsonl(path: str, item: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_prediction_en(answer: Any, m: int) -> List[int]:
    try:
        predicted = json.loads(answer)["little_penguin"]
    except (json.JSONDecodeError, KeyError, TypeError):
        predicted = extract_numbers_from_string(str(answer))
    return reduce_duplicate(predicted or [], m)


def parse_prediction_zh(answer: Any, m: int) -> List[int]:
    try:
        if isinstance(answer, str) and "```" in answer:
            predicted = json_repair.loads(answer.replace("```", "").replace("json", "").strip())["小企鹅"]
        elif isinstance(answer, str):
            predicted = json_repair.loads(answer)["小企鹅"]
        else:
            predicted = answer["小企鹅"]
    except Exception:
        predicted = extract_numbers_from_string(str(answer))
    return reduce_duplicate(predicted or [], m)


def score_jsonl(
    file_obj,
    language: str,
    max_context_length: int,
    m: int,
    n: int,
    test_type: str,
    a_stars: List[int],
    r_stars: List[int],
) -> Tuple[pd.DataFrame, float]:
    if language == "EN":
        scalar = 0.82 if test_type == "Acquisition" else 0.815
    elif language == "ZH":
        scalar = 0.725 if test_type == "Acquisition" else 0.72
    else:
        raise ValueError(f"Unknown language: {language}")

    data = []
    for item in iter_jsonl(file_obj):
        answer = item["answer"]
        predicted = parse_prediction_en(answer, m) if language == "EN" else parse_prediction_zh(answer, m)
        for i in range(1, m + 1):
            if test_type == "Acquisition":
                try:
                    score = 1.0 if item["reference_counting_results"][i - 1] in predicted else 0.0
                except Exception:
                    score = 0.0
            else:
                score = get_reasoning_score(i - 1, predicted, a_stars, r_stars)
            data.append({
                "Counting Times": i,
                "Context Size": int(item["context_size"] / scalar),
                "Score": score,
            })

    df = pd.DataFrame(data)
    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Counting Times", "Context Size"],
        aggfunc="mean",
    ).reset_index()
    pivot_table = pivot_table.pivot(
        index="Counting Times",
        columns="Context Size",
        values="Score",
    )
    return pivot_table, float(pivot_table.mean(axis=None).round(3))


def save_counting_heatmap(pivot_table: pd.DataFrame, mean_score: float, test_type: str, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [
            "#184E77", "#1E6091", "#1A759F", "#168AAD", "#34A0A4",
            "#52B69A", "#76C893", "#99D98C", "#B5E48C", "#D9ED92",
        ],
    )

    fig = plt.figure(figsize=(12, 6))
    heatmap = sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        linewidths=1,
        cbar_kws={"label": "Score", "pad": 0.03},
        vmin=0,
        vmax=1,
    )

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Score", fontsize=20)

    plt.title(f"Llama-2-7B {test_type}: score={mean_score}", size=22, pad=16, fontweight="bold")
    plt.xlabel("Context Length", size=20)
    plt.ylabel("Counting Times", size=20)
    plt.xticks(rotation=45, size=12)
    plt.yticks(rotation=0, size=12)
    fig.subplots_adjust(left=0.12, right=0.9, bottom=0.18, top=0.84)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {test_type} heatmap saved to: {save_path}", flush=True)


def normalize_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    config = {
        "rodrope_block": RODROPE_BLOCK,
        "m1": M1,
        "lambda2": LAMBDA2,
        "m2": M2,
        "lambda3": LAMBDA3,
        "m3": M3,
        "lambda4": LAMBDA4,
    }
    config.update(raw_config)

    rodrope_block = int(config["rodrope_block"])
    if rodrope_block not in (2, 3, 4):
        raise ValueError("rodrope_block must be 2, 3 or 4.")

    out = {
        "rodrope_block": rodrope_block,
        "m1": int(config["m1"]),
        "lambda2": float(config["lambda2"]),
    }
    if rodrope_block == 2:
        out["m2"] = None
        out["lambda3"] = None
        out["m3"] = None
        out["lambda4"] = None
        return out

    if config.get("m2") is None or config.get("lambda3") is None:
        raise ValueError("block>=3 requires m2 and lambda3.")
    if int(config["m2"]) <= int(config["m1"]):
        raise ValueError("For block>=3, m2 must be larger than m1.")
    out["m2"] = int(config["m2"])
    out["lambda3"] = float(config["lambda3"])

    if rodrope_block == 3:
        out["m3"] = None
        out["lambda4"] = None
        return out

    if config.get("m3") is None or config.get("lambda4") is None:
        raise ValueError("block=4 requires m3 and lambda4.")
    if int(config["m3"]) <= int(config["m2"]):
        raise ValueError("For block=4, m3 must be larger than m2.")
    out["m3"] = int(config["m3"])
    out["lambda4"] = float(config["lambda4"])
    return out


def load_model_and_tokenizer(model_path: str, config: Dict[str, Any], flash_impl: str):
    print(f"[load] loading model from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    patch_rope_for_rodrope(model)
    apply_kwargs = dict(
        group_size=config["lambda2"],
        window_size=config["m1"],
        enable_flash_attention=True,
        flash_attention_impl=flash_impl,
        block=config["rodrope_block"],
    )
    if config["rodrope_block"] >= 3:
        apply_kwargs.update(
            far_size=config["m2"],
            far_group_size=config["lambda3"],
        )
    if config["rodrope_block"] == 4:
        apply_kwargs.update(
            far2_size=config["m3"],
            far2_group_size=config["lambda4"],
        )
    Rodrope.apply(model, **apply_kwargs)

    print(
        f"[Rodrope] block={config['rodrope_block']}, "
        f"m1={config['m1']}, lambda2={config['lambda2']}, "
        f"m2={config['m2']}, lambda3={config['lambda3']}, "
        f"m3={config['m3']}, lambda4={config['lambda4']}",
        flush=True,
    )
    return model, tokenizer


def cleanup_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            with torch.cuda.device(dev_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def run_counting_eval(
    config: Dict[str, Any],
    language: str = LANGUAGE,
    m: int = M,
    n: int = N,
    max_context_length: int = MAX_CONTEXT_LENGTH,
    model_path: str = MODEL_PATH,
    results_root: str = RESULTS_ROOT,
    run_tag: str = RUN_TAG,
    flash_impl: str = RODROPE_FLASH_IMPL,
) -> Dict[str, Any]:
    os.makedirs(results_root, exist_ok=True)
    config = normalize_config(config)

    lambda2_tag = format_float_for_filename(config["lambda2"])
    lambda3_tag = format_float_for_filename(config["lambda3"]) if config["rodrope_block"] >= 3 else "none"
    lambda4_tag = format_float_for_filename(config["lambda4"]) if config["rodrope_block"] == 4 else "none"
    m2_tag = str(config["m2"]) if config["rodrope_block"] >= 3 else "none"
    m3_tag = str(config["m3"]) if config["rodrope_block"] == 4 else "none"
    base_uid = uuid.uuid4().hex[:8]

    print(
        f"[counting] run_tag={run_tag}, language={language}, m={m}, n={n}, "
        f"max_context_length={max_context_length}",
        flush=True,
    )

    model, tokenizer = load_model_and_tokenizer(model_path, config, flash_impl)

    a_stars = get_stars(os.path.join(COUNTING_CONTEXT_ROOT, "a_stars.txt"), m)
    r_stars = get_stars(os.path.join(COUNTING_CONTEXT_ROOT, "r_stars.txt"), m)
    context = get_context(language)

    mean_scores: Dict[str, float] = {}
    output_files: Dict[str, str] = {}
    visualization_files: Dict[str, str] = {}

    for test_type in ["acquisition", "reasoning"]:
        testing_type = test_type.capitalize()
        tmp_tag = (
            f"lang={language}"
            f"_type={test_type}"
            f"_m={m}"
            f"_n={n}"
            f"_maxlen={max_context_length}"
            f"_block={config['rodrope_block']}"
            f"_m1={config['m1']}"
            f"_lambda2={lambda2_tag}"
            f"_m2={m2_tag}"
            f"_lambda3={lambda3_tag}"
            f"_m3={m3_tag}"
            f"_lambda4={lambda4_tag}"
            f"_uid={base_uid}"
        )
        output_file = os.path.join(results_root, f"{run_tag}_result_Rodrope_flash_{tmp_tag}.jsonl")
        if os.path.exists(output_file):
            os.remove(output_file)

        if language == "EN":
            scalar = 0.82 if testing_type == "Acquisition" else 0.815
        else:
            scalar = 0.725 if testing_type == "Acquisition" else 0.72

        interval = int(max_context_length / n)
        context_size_list = [int(i * scalar) for i in range(interval, max_context_length + 1, interval)]
        print(f"[counting] {test_type}: context sizes = {context_size_list}", flush=True)

        for file_id, j in enumerate(context_size_list, start=1):
            indicator = 0
            sprinkle_stars_context = " ".join(context.split(" ")[:j])

            for k in range(0, j, int(j / m)):
                single_star = sentence_with_star(language, test_type, indicator, a_stars, r_stars)
                sprinkle_words = sprinkle_stars_context.split(" ")
                single_star_words = single_star.split(" ")
                insert_start = len(single_star_words) * indicator + k + int(j / m)
                insert_end = int(j / m) + k + len(single_star_words) * indicator
                sprinkle_stars_context = (
                    " ".join(sprinkle_words[:insert_start])
                    + single_star
                    + " ".join(sprinkle_words[insert_end:])
                )
                indicator += 1
                if indicator == m:
                    break

            retrieval_question = select_question(language, test_type)
            prompt_text = generate_prompt(sprinkle_stars_context, retrieval_question)
            enc = tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            result = {
                "context_size": j,
                "answer": response,
                "reference_counting_results": a_stars if test_type == "acquisition" else r_stars,
            }
            append_jsonl(output_file, result)
            print(f"[counting] {test_type} #{file_id}/{len(context_size_list)} context_size={j} answer={response}", flush=True)

        with open(output_file, "r", encoding="utf-8") as f:
            pivot_table, mean_score = score_jsonl(
                f,
                language=language,
                max_context_length=max_context_length,
                m=m,
                n=n,
                test_type=testing_type,
                a_stars=a_stars,
                r_stars=r_stars,
            )

        mean_scores[test_type] = mean_score
        output_files[test_type] = output_file
        viz_path = ACQ_PDF_PATH if test_type == "acquisition" else REA_PDF_PATH
        save_counting_heatmap(pivot_table, mean_score, testing_type, viz_path)
        visualization_files[test_type] = viz_path
        print(f"[counting] {test_type} mean_score = {mean_score}", flush=True)

    cleanup_model(model, tokenizer)

    return {
        "config": config,
        "language": language,
        "m": m,
        "n": n,
        "max_context_length": max_context_length,
        "acquisition": mean_scores["acquisition"],
        "reasoning": mean_scores["reasoning"],
        "output_files": output_files,
        "visualization_files": visualization_files,
    }


def read_configs_arg(configs_arg: Optional[str]):
    if not configs_arg:
        return None
    if os.path.exists(configs_arg):
        with open(configs_arg, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(configs_arg)


def parse_args():
    parser = argparse.ArgumentParser(description="Counting-Stars acquisition/reasoning evaluation for Llama-2 + Rodrope; no PPL.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--results_root", type=str, default=RESULTS_ROOT)
    parser.add_argument("--language", type=str, default=LANGUAGE, choices=["EN", "ZH"])
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--max_context_length", type=int, default=MAX_CONTEXT_LENGTH)
    parser.add_argument("--run_tag", type=str, default=RUN_TAG)
    parser.add_argument("--rodrope_block", type=int, default=RODROPE_BLOCK, choices=[2, 3, 4])
    parser.add_argument("--m1", type=int, default=M1)
    parser.add_argument("--lambda2", type=float, default=LAMBDA2)
    parser.add_argument("--m2", type=int, default=M2)
    parser.add_argument("--lambda3", type=float, default=LAMBDA3)
    parser.add_argument("--m3", type=int, default=M3)
    parser.add_argument("--lambda4", type=float, default=LAMBDA4)
    parser.add_argument("--rodrope_flash_impl", type=str, default=RODROPE_FLASH_IMPL, choices=["flash_attn", "triton"])
    parser.add_argument("--configs", type=str, default=None, help="JSON string or JSON file path with one config dict or a list of config dicts.")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_configs = read_configs_arg(args.configs)
    if raw_configs is None and COUNTING_CONFIGS:
        raw_configs = COUNTING_CONFIGS
    if raw_configs is None:
        raw_configs = {
            "rodrope_block": args.rodrope_block,
            "m1": args.m1,
            "lambda2": args.lambda2,
            "m2": args.m2,
            "lambda3": args.lambda3,
            "m3": args.m3,
            "lambda4": args.lambda4,
        }
    if isinstance(raw_configs, dict):
        raw_configs = [raw_configs]

    all_results = []
    for index, raw_config in enumerate(raw_configs, start=1):
        print(f"\n[main] evaluating config {index}/{len(raw_configs)}", flush=True)
        result = run_counting_eval(
            raw_config,
            language=args.language,
            m=args.m,
            n=args.n,
            max_context_length=args.max_context_length,
            model_path=args.model_path,
            results_root=args.results_root,
            run_tag=f"{args.run_tag}_cfg{index}" if len(raw_configs) > 1 else args.run_tag,
            flash_impl=args.rodrope_flash_impl,
        )
        all_results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print(
            f"[result] config {index}: acquisition={result['acquisition']}, reasoning={result['reasoning']}",
            flush=True,
        )

    if all_results:
        print("\n[final results]", flush=True)
        for index, result in enumerate(all_results, start=1):
            print(
                f"config {index}: acquisition={result['acquisition']}, reasoning={result['reasoning']}",
                flush=True,
            )

    summary_path = os.path.join(args.results_root, f"{args.run_tag}_counting_summary.json")
    os.makedirs(args.results_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[done] summary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

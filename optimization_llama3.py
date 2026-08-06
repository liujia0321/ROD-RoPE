# -*- coding: utf-8 -*-
# transformers==4.38.2
# 需要: botorch, gpytorch, json_repair, pandas, transformers, torch
import os
import sys
import gc
import uuid
import json
import ast
import glob
import random
import re
import subprocess
import argparse
import warnings
from typing import Any, List, Tuple
import time
import csv
import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/home/liujia/allcode/meta-llamaMeta-Llama-3-8B-Instruct"
RESULTS_ROOT = "/home/liujia/allcode/ROPE-MLP/counting-stars-jsonl3-allblock"
COUNTING_CONTEXT_ROOT = "/home/liujia/allcode/Counting-Stars-main/context_data"
LANGUAGE = "EN"
M = 8
N = 8
MAX_CONTEXT_LENGTH = 16768
RODROPE_FLASH_IMPL = "flash_attn"

import torch
from torch import nn, Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import pandas as pd
import json_repair

# =======================
# Rodrope 库
# =======================
import Rodrope  # 你已有的 Rodrope 库

# =======================
# BoTorch / GP 相关
# =======================
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

try:
    from botorch.acquisition.analytic import ExpectedImprovement
except ImportError:
    from botorch.acquisition import ExpectedImprovement

from botorch.optim.optimize import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

# =====================================================================
# 1. RoPE 适配: RodropeRopeAdapter + patch_rope_for_rodrope
# =====================================================================

class RodropeRopeAdapter(nn.Module):
    """
    包一层 LlamaRotaryEmbedding，使其兼容 Rodrope 内部
    调用形式 rope(seq_len=..., device=..., dtype=...).
    """
    def __init__(self, rope):
        super().__init__()
        self.rope = rope  # 原始的 rotary_emb 模块

    def forward(self, *args, **kwargs):
        # Rodrope 调用时：rotary_emb(seq_len=..., device=..., dtype=...)
        if "seq_len" in kwargs and len(args) == 0:
            seq_len = kwargs.pop("seq_len")
            device = kwargs.pop("device", None)
            dtype = kwargs.pop("dtype", None)

            try:
                self.rope._set_cos_sin_cache(seq_len, device=device, dtype=dtype)
            except TypeError:
                self.rope._set_cos_sin_cache(seq_len, device, dtype)
            return (self.rope.cos_cached, self.rope.sin_cached)

        return self.rope(*args, **kwargs)

    def __getattr__(self, name):
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

# =====================================================================
# 2. Counting-Stars：上下文、星星、打分工具
# =====================================================================

def get_context(language: str, context_root: str = COUNTING_CONTEXT_ROOT) -> str:
    if language == "EN":
        context = ""
        pattern = os.path.join(context_root, "PaulGrahamEssays", "*.txt")
        for file in glob.glob(pattern):
            with open(file, "r", encoding="utf-8") as f:
                context += f.read().replace("\n", " ")
        return context
    elif language == "ZH":
        string_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        with open(
            os.path.join(context_root, "The_Story_of_the_Stone.txt"),
            "r",
            encoding="utf-8",
        ) as context_file:
            context = ""
            for i in context_file.readlines():
                context += i.strip().replace("------------", " ").replace("\n", " ").replace(" ", "")
        context = re.sub('[{}]'.format(string_punctuation), "", context)
        context = re.sub('[a-zA-Z]', '', context)
        return context
    else:
        raise ValueError(f"Unknown language: {language}")


def get_stars(stars_dir: str, counting_times: int) -> List[int]:
    with open(stars_dir, "r", encoding="utf-8") as stars_file:
        stars_data = ast.literal_eval(stars_file.readline())
    return [int(value) for value in stars_data[str(counting_times)]]


def sentence_with_star(language, test_type, indicator, a_stars, r_stars):
    tt = test_type.lower()
    if language == "ZH":
        if tt == "acquisition":
            single_star = f"\n小企鹅数了{a_stars[indicator]}颗★\n"
        elif tt == "reasoning":
            single_star = f"\n小企鹅数了{r_stars[indicator]}颗★，但发现数错了，于是又数了一遍，这次数对了，是{a_stars[indicator]}颗★\n"
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
    elif language == "EN":
        if tt == "acquisition":
            single_star = f"\nThe little penguin counted {a_stars[indicator]} ★\n"
        elif tt == "reasoning":
            single_star = (
                f"\nThe little penguin counted {r_stars[indicator]} ★, "
                f"but found that a mistake had been made, so the counting was done again, "
                f"and this time {a_stars[indicator]} ★ was counted correctly.\n"
            )
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
    else:
        raise ValueError(f"Unknown language: {language}")
    return single_star


def select_question(language, test_type):
    tt = test_type.lower()
    if language == "ZH":
        acquisition_question = (
            "\n\n\n\n在这个月光皎洁、云雾缭绕的夜晚，小企鹅正望向天空，全神贯注地数★。"
            "请帮助小企鹅收集所数★的颗数，按照如下格式：{\"小企鹅\":[x,x,x,...]}，不要求和，"
            "[x,x,x,...]中数字为小企鹅每次数★的颗数，仅以JSON格式输出结果，不需要输出任何解释。"
        )
        reasoning_question = (
            "\n\n\n\n在这个月光皎洁、云雾缭绕的夜晚，小企鹅正望向天空，全神贯注地数★。"
            "请帮助小企鹅收集所数★的正确颗数，按照如下格式：{\"小企鹅\":[x,x,x,...]}，不要求和，"
            "[x,x,x,...]中数字为小企鹅正确数★的颗数，仅以JSON格式输出结果，不需要输出任何解释。"
        )
        if tt == "acquisition":
            return acquisition_question
        elif tt == "reasoning":
            return reasoning_question
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

    elif language == "EN":
        acquisition_question = (
            "\n\n\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. "
            "Please help the little penguin collect the number of ★, for example: {\"little_penguin\": [x, x, x,...]}. "
            "The summation is not required, and the numbers in [x, x, x,...] represent the counted number of ★ by the little penguin. "
            "Only output the results in JSON format without any explanation."
        )
        reasoning_question = (
            "\n\n\n\nOn this moonlit and misty night, the little penguin is looking up at the sky and concentrating on counting ★. "
            "Please help the little penguin collect the correct number of ★, for example: {\"little_penguin\": [x, x, x,...]}. "
            "The summation is not required, and the numbers in [x, x, x,...] represent the correctly counted number of ★ by the little penguin. "
            "Only output the results in JSON format without any explanation."
        )
        if tt == "acquisition":
            return acquisition_question
        elif tt == "reasoning":
            return reasoning_question
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
    else:
        raise ValueError(f"Unknown language: {language}")


def generate_prompt(context: str, retrieval_question: str) -> str:
    test_format = (
        f" This is a very long story book: <book> {context} </book>.\n"
        f" Based on the content of the book, Question: {retrieval_question}\nAnswer:"
    )
    return test_format


# =======================
# 打分工具
# =======================

def extract_numbers_from_string(string: str) -> List[int]:
    numbers = re.findall(r'\d+', string)
    return [int(num) for num in numbers] if numbers else []


def get_reasoning_score(index: int, predicted: List[int], a_stars: List[int], r_stars: List[int]) -> float:
    if a_stars[index] in predicted and r_stars[index] in predicted:
        return 0.5
    elif a_stars[index] in predicted and r_stars[index] not in predicted:
        return 1.0
    elif a_stars[index] not in predicted and r_stars[index] in predicted:
        return 0.25
    else:
        return 0.0


def get_context_size(max_context_length, n):
    intervel = int(max_context_length / n)
    return [i for i in range(intervel, max_context_length + 1, intervel)]


def reduce_duplicate(predicted: Any, m: int) -> List[int]:
    if not isinstance(predicted, (list, tuple, set)):
        predicted = extract_numbers_from_string(str(predicted))

    reduced = []
    seen = set()
    for value in predicted:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in seen:
            reduced.append(number)
            seen.add(number)
        if len(reduced) == m:
            break
    return reduced


def iter_jsonl(file_obj):
    for line in file_obj:
        line = line.strip()
        if line:
            yield json.loads(line)


def append_jsonl(path: str, item: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def strip_json_fence(answer: Any) -> str:
    text = str(answer).strip()
    if "```" not in text:
        return text
    return text.replace("```", "").replace("json", "").strip()


def parse_prediction_en(answer: Any, m: int) -> List[int]:
    try:
        predicted = json.loads(strip_json_fence(answer))["little_penguin"]
    except (json.JSONDecodeError, KeyError, TypeError):
        try:
            predicted = json_repair.loads(strip_json_fence(answer))["little_penguin"]
        except Exception:
            predicted = extract_numbers_from_string(str(answer))
    return reduce_duplicate(predicted or [], m)


def parse_prediction_zh(answer: Any, m: int) -> List[int]:
    try:
        if isinstance(answer, str):
            predicted = json_repair.loads(strip_json_fence(answer))["小企鹅"]
        else:
            predicted = answer["小企鹅"]
    except Exception:
        predicted = extract_numbers_from_string(str(answer))
    return reduce_duplicate(predicted or [], m)


def get_data_EN(folder_path, max_context_length, m, n, test_type, a_stars, r_stars):
    data = []
    if test_type == "Acquisition":
        scalar = 0.82
    elif test_type == "Reasoning":
        scalar = 0.815
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    for item in iter_jsonl(folder_path):
        predicted = parse_prediction_en(item["answer"], m)
        for i in range(1, m + 1):
            counting_times = i
            if test_type == "Acquisition":
                try:
                    score = 1.0 if item["reference_counting_results"][i - 1] in predicted else 0.0
                except Exception:
                    score = 0.0
            else:
                score = get_reasoning_score(counting_times - 1, predicted, a_stars, r_stars)
            data.append({
                "Counting Times": counting_times,
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
    return pivot_table, pivot_table.mean(axis=None).round(3)


def get_data_ZH(folder_path, max_context_length, m, n, test_type, a_stars, r_stars):
    data = []
    if test_type == "Acquisition":
        scalar = 0.725
    elif test_type == "Reasoning":
        scalar = 0.72
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    for item in iter_jsonl(folder_path):
        predicted = parse_prediction_zh(item["answer"], m)
        for i in range(1, m + 1):
            counting_times = i
            if test_type == "Acquisition":
                try:
                    score = 1.0 if item["reference_counting_results"][i - 1] in predicted else 0.0
                except Exception:
                    score = 0.0
            else:
                score = get_reasoning_score(counting_times - 1, predicted, a_stars, r_stars)
            data.append({
                "Counting Times": counting_times,
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
    return pivot_table, pivot_table.mean(axis=None).round(3)


# =====================================================================
# 3. worker 子进程：给定固定 block 的 Rodrope 参数，评估 acquisition + reasoning
# =====================================================================

def format_float_for_filename(value: float, digits: int = 6) -> str:
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def worker_eval_both(
    rodrope_block: int,
    m1: int,
    lambda2: float,
    m2: int = 0,
    lambda3: float = 0.0,
    m3: int = 0,
    lambda4: float = 0.0,
    language: str = LANGUAGE,
    m: int = M,
    n: int = N,
    max_context_length: int = MAX_CONTEXT_LENGTH,
    model_path: str = MODEL_PATH,
    results_root: str = RESULTS_ROOT,
    counting_context_root: str = COUNTING_CONTEXT_ROOT,
    run_tag: str = "manual",
    flash_impl: str = RODROPE_FLASH_IMPL,
) -> Tuple[float, float]:
    """
    子进程里调用：
      - 加载一次模型 + Rodrope
      - block=2: 使用 (m1, lambda2)
      - block=3: 额外使用 (m2, lambda3)
      - block=4: 额外使用 (m3, lambda4)
      - 对 test_type in {"acquisition", "reasoning"} 各跑一遍
      - 返回 (mean_acq, mean_reason)
    """

    rodrope_block = int(rodrope_block)
    m1 = int(m1)
    m2 = int(m2) if rodrope_block >= 3 else 0
    m3 = int(m3) if rodrope_block == 4 else 0

    if rodrope_block not in (2, 3, 4):
        raise ValueError("rodrope_block must be 2, 3 or 4.")
    if rodrope_block >= 3 and m2 <= m1:
        raise ValueError("For block>=3, m2 must be larger than m1.")
    if rodrope_block == 4 and m3 <= m2:
        raise ValueError("For block=4, m3 must be larger than m2.")

    os.makedirs(results_root, exist_ok=True)

    lambda2_raw = float(lambda2)
    lambda3_raw = float(lambda3) if rodrope_block >= 3 else 0.0
    lambda4_raw = float(lambda4) if rodrope_block == 4 else 0.0
    lambda2_tag = format_float_for_filename(lambda2_raw)
    lambda3_tag = format_float_for_filename(lambda3_raw) if rodrope_block >= 3 else "none"
    lambda4_tag = format_float_for_filename(lambda4_raw) if rodrope_block == 4 else "none"
    m2_tag = str(m2) if rodrope_block >= 3 else "none"
    m3_tag = str(m3) if rodrope_block == 4 else "none"
    run_tag = (run_tag or "manual").strip()

    print(
        f"[worker] start eval: run_tag={run_tag}, block={rodrope_block}, "
        f"m1={m1}, lambda2={lambda2_raw}, "
        f"m2={m2_tag}, lambda3={lambda3_tag}, "
        f"m3={m3_tag}, lambda4={lambda4_tag}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(model_path)
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        print("[load] success with attn_implementation=flash_attention_2", flush=True)
    except Exception as e:
        print(f"[warn] flash_attention_2 load failed: {e}\n       fallback to eager.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            attn_implementation="eager",
            device_map="auto",
        )
        print("[load] success with attn_implementation=eager", flush=True)
    model.eval()

    patch_rope_for_rodrope(model)
    apply_kwargs = dict(
        lambda2=lambda2_raw,
        m1=m1,
        enable_flash_attention=True,
        flash_attention_impl=flash_impl,
        rodrope_block=rodrope_block,
    )
    if rodrope_block >= 3:
        apply_kwargs.update(
            m2=m2,
            lambda3=lambda3_raw,
        )
    if rodrope_block == 4:
        apply_kwargs.update(
            m3=m3,
            lambda4=lambda4_raw,
        )
    Rodrope.apply(model, **apply_kwargs)

    a_stars = get_stars(os.path.join(counting_context_root, "a_stars.txt"), m)
    r_stars = get_stars(os.path.join(counting_context_root, "r_stars.txt"), m)
    context = get_context(language, counting_context_root)

    base_uid = uuid.uuid4().hex[:8]
    mean_scores = {}

    for test_type in ["acquisition", "reasoning"]:
        testing_type = test_type.capitalize()
        tmp_tag = (
            f"lang={language}"
            f"_type={test_type}"
            f"_m={m}"
            f"_n={n}"
            f"_maxlen={max_context_length}"
            f"_block={rodrope_block}"
            f"_m1={m1}"
            f"_lambda2={lambda2_tag}"
            f"_m2={m2_tag}"
            f"_lambda3={lambda3_tag}"
            f"_m3={m3_tag}"
            f"_lambda4={lambda4_tag}"
            f"_uid={base_uid}"
        )
        output_file = os.path.join(
            results_root,
            f"{run_tag}_result_Rodrope_flash_{tmp_tag}.jsonl"
        )

        if os.path.exists(output_file):
            os.remove(output_file)

        if language == "EN":
            scalar = 0.82 if testing_type == "Acquisition" else 0.815
        else:
            scalar = 0.725 if testing_type == "Acquisition" else 0.72

        version = [[m, n]]

        for m_, n_ in version:
            interval = int(max_context_length / n_)
            context_size_list = [
                int(i * scalar) for i in range(interval, max_context_length + 1, interval)
            ]

            for j in context_size_list:
                indicator = 0
                sprinkle_stars_context = " ".join(context.split(" ")[:j])

                for k in range(0, j, int(j / m_)):
                    single_star = sentence_with_star(
                        language, test_type, indicator, a_stars, r_stars
                    )
                    sprinkle_stars_context = (
                        " ".join(
                            sprinkle_stars_context.split(" ")[
                                :len(single_star.split(" ")) * indicator
                                 + k
                                 + int(j / m_)
                            ]
                        )
                        + single_star
                        + " ".join(
                            sprinkle_stars_context.split(" ")[
                                int(j / m_)
                                + k
                                + len(single_star.split(" ")) * indicator:
                            ]
                        )
                    )
                    indicator += 1
                    if indicator == m_:
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
                response = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

                result = {
                    "context_size": j,
                    "answer": response,
                    "reference_counting_results": (
                        a_stars if test_type == "acquisition" else r_stars
                    ),
                }
                append_jsonl(output_file, result)

        if language == "EN":
            with open(output_file, "r", encoding="utf-8") as f:
                _, mean_score = get_data_EN(
                    f,
                    max_context_length=max_context_length,
                    m=m,
                    n=n,
                    test_type=testing_type,
                    a_stars=a_stars,
                    r_stars=r_stars,
                )
        else:
            with open(output_file, "r", encoding="utf-8") as f:
                _, mean_score = get_data_ZH(
                    f,
                    max_context_length=max_context_length,
                    m=m,
                    n=n,
                    test_type=testing_type,
                    a_stars=a_stars,
                    r_stars=r_stars,
                )

        mean_scores[test_type] = float(mean_score)
        print(f"[worker] {test_type} mean_score = {mean_score}", flush=True)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            with torch.cuda.device(dev_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    return mean_scores["acquisition"], mean_scores["reasoning"]


def worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="worker mode")
    parser.add_argument("--rodrope_block", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--m1", type=int, required=True)
    parser.add_argument("--lambda2", type=float, required=True)
    parser.add_argument("--m2", type=int, default=0)
    parser.add_argument("--lambda3", type=float, default=0.0)
    parser.add_argument("--m3", type=int, default=0)
    parser.add_argument("--lambda4", type=float, default=0.0)
    parser.add_argument("--language", type=str, default=LANGUAGE, choices=["EN", "ZH"])
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--max_context_length", type=int, default=MAX_CONTEXT_LENGTH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--results_root", type=str, default=RESULTS_ROOT)
    parser.add_argument("--counting_context_root", type=str, default=COUNTING_CONTEXT_ROOT)
    parser.add_argument("--rodrope_flash_impl", type=str, default=RODROPE_FLASH_IMPL, choices=["flash_attn", "triton"])
    parser.add_argument("--run_tag", type=str, default="manual")
    args = parser.parse_args()

    f_acq, f_reason = worker_eval_both(
        rodrope_block=args.rodrope_block,
        m1=args.m1,
        lambda2=args.lambda2,
        m2=args.m2,
        lambda3=args.lambda3,
        m3=args.m3,
        lambda4=args.lambda4,
        language=args.language,
        m=args.m,
        n=args.n,
        max_context_length=args.max_context_length,
        model_path=args.model_path,
        results_root=args.results_root,
        counting_context_root=args.counting_context_root,
        run_tag=args.run_tag,
        flash_impl=args.rodrope_flash_impl,
    )
    print(json.dumps({"f_acq": f_acq, "f_reason": f_reason}))


EXPERIMENT_DIR = RESULTS_ROOT
LOG_FILE = os.path.join(EXPERIMENT_DIR, "llama3_allblock_multiotimization_bo_2objective_log.csv")
GLOBAL_EVAL_ID = 0


def set_experiment_dir(experiment_dir: str) -> None:
    global EXPERIMENT_DIR, LOG_FILE
    EXPERIMENT_DIR = experiment_dir
    LOG_FILE = os.path.join(EXPERIMENT_DIR, "llama3_allblock_multiotimization_bo_2objective_log.csv")


def init_log_file():
    """Create a log file with raw and normalized objectives."""
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    expected_header = [
        "timestamp",
        "stage",
        "outer_iter",
        "init_index",
        "eval_id",
        "rodrope_block",
        "m1",
        "lambda2",
        "m2",
        "lambda3",
        "m3",
        "lambda4",
        "f_acq",
        "f_reason",
        "f_total",
        "norm_f_acq",
        "norm_f_reason",
        "norm_f_total",
        "elapsed_sec",
    ]
    rewrite = True
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", newline="") as f:
            first = f.readline().strip().split(",")
        rewrite = first != expected_header
    if rewrite:
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(expected_header)


def log_eval_result(
    stage: str,
    outer_iter: int,
    init_index: int,
    rodrope_block: int,
    m1: float,
    lambda2: float,
    m2: float,
    lambda3: float,
    m3: float,
    lambda4: float,
    f_acq: float,
    f_reason: float,
    elapsed_sec: float,
):
    """向日志文件追加一行记录。"""
    global GLOBAL_EVAL_ID
    GLOBAL_EVAL_ID += 1

    ts = datetime.datetime.now().isoformat(timespec="seconds")

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            stage,
            outer_iter,
            init_index,
            GLOBAL_EVAL_ID,
            int(rodrope_block),
            float(m1),
            float(lambda2),
            float(m2),
            float(lambda3),
            float(m3),
            float(lambda4),
            float(f_acq),
            float(f_reason),
            float(f_acq) + float(f_reason),
            float(f_acq),
            float(f_reason),
            float(f_acq) + float(f_reason),
            float(elapsed_sec),
        ])

# =====================================================================
# 5. Fixed-block BO 主过程（在主进程里跑）
# =====================================================================

M1_VALUES = torch.tensor(
    [128, 256, 512, 1024, 2048, 3072, 4096],
    dtype=torch.float64,
)
M2_VALUES = torch.tensor(
    [1024, 2048, 4096, 8192, 12288, 16384, 24576],
    dtype=torch.float64,
)
M3_VALUES = M2_VALUES.clone()
NUM_M1 = M1_VALUES.numel()
NUM_M2 = M2_VALUES.numel()
NUM_M3 = M3_VALUES.numel()
LAMBDA2_MIN = 4.0
LAMBDA2_MAX = 32.0
LAMBDA3_MIN = 4.0
LAMBDA3_MAX = 32.0
LAMBDA4_MIN = 4.0
LAMBDA4_MAX = 32.0
INITIAL_SAMPLE_SIZE = 10
INNER_ITERATIONS = 50
EI_XI = 0.05
GP_NOISE_VARIANCE = 1e-6
M12IDX = {int(v.item()): i for i, v in enumerate(M1_VALUES)}
M22IDX = {int(v.item()): i for i, v in enumerate(M2_VALUES)}
M32IDX = {int(v.item()): i for i, v in enumerate(M3_VALUES)}


def valid_discrete_configs_for_block(rodrope_block: int):
    configs = []
    if rodrope_block == 2:
        for window_val in M1_VALUES:
            configs.append((float(window_val.item()), 0.0, 0.0))
    elif rodrope_block == 3:
        for window_val in M1_VALUES:
            for far_val in M2_VALUES:
                if int(far_val.item()) > int(window_val.item()):
                    configs.append((float(window_val.item()), float(far_val.item()), 0.0))
    elif rodrope_block == 4:
        for window_val in M1_VALUES:
            for far_val in M2_VALUES:
                if int(far_val.item()) <= int(window_val.item()):
                    continue
                for far2_val in M3_VALUES:
                    if int(far2_val.item()) > int(far_val.item()):
                        configs.append((float(window_val.item()), float(far_val.item()), float(far2_val.item())))
    else:
        raise ValueError("rodrope_block must be 2, 3 or 4.")
    if not configs:
        raise ValueError(f"No valid discrete configs for block={rodrope_block}.")
    return configs


def encode_lambda2(lambda2: float) -> float:
    return (lambda2 - LAMBDA2_MIN) / (LAMBDA2_MAX - LAMBDA2_MIN)


def decode_lambda2(lambda2_norm: float) -> float:
    return LAMBDA2_MIN + lambda2_norm * (LAMBDA2_MAX - LAMBDA2_MIN)


def encode_lambda3(lambda3: float) -> float:
    return (lambda3 - LAMBDA3_MIN) / (LAMBDA3_MAX - LAMBDA3_MIN)


def decode_lambda3(lambda3_norm: float) -> float:
    return LAMBDA3_MIN + lambda3_norm * (LAMBDA3_MAX - LAMBDA3_MIN)


def encode_lambda4(lambda4: float) -> float:
    return (lambda4 - LAMBDA4_MIN) / (LAMBDA4_MAX - LAMBDA4_MIN)


def decode_lambda4(lambda4_norm: float) -> float:
    return LAMBDA4_MIN + lambda4_norm * (LAMBDA4_MAX - LAMBDA4_MIN)


def encode_m1(m1: float) -> float:
    idx = M12IDX[int(m1)]
    if NUM_M1 == 1:
        return 0.0
    return idx / (NUM_M1 - 1)


def decode_m1(m1_norm: float) -> float:
    if NUM_M1 == 1:
        return float(M1_VALUES[0].item())
    idx = int(round(m1_norm * (NUM_M1 - 1)))
    idx = max(0, min(NUM_M1 - 1, idx))
    return float(M1_VALUES[idx].item())


def encode_m2(m2: float) -> float:
    if float(m2) <= 0:
        return 0.0
    idx = M22IDX[int(m2)]
    if NUM_M2 == 1:
        return 0.0
    return idx / (NUM_M2 - 1)


def decode_m2(m2_norm: float) -> float:
    if NUM_M2 == 1:
        return float(M2_VALUES[0].item())
    idx = int(round(m2_norm * (NUM_M2 - 1)))
    idx = max(0, min(NUM_M2 - 1, idx))
    return float(M2_VALUES[idx].item())


def encode_m3(m3: float) -> float:
    if float(m3) <= 0:
        return 0.0
    idx = M32IDX[int(m3)]
    if NUM_M3 == 1:
        return 0.0
    return idx / (NUM_M3 - 1)


def decode_m3(m3_norm: float) -> float:
    if NUM_M3 == 1:
        return float(M3_VALUES[0].item())
    idx = int(round(m3_norm * (NUM_M3 - 1)))
    idx = max(0, min(NUM_M3 - 1, idx))
    return float(M3_VALUES[idx].item())


def encode_config(
    rodrope_block: int,
    m1: float,
    lambda2: float,
    m2: float = 0.0,
    lambda3: float = 0.0,
    m3: float = 0.0,
    lambda4: float = 0.0,
) -> Tensor:
    if int(rodrope_block) == 2:
        m2 = 0.0
        lambda3 = 0.0
        m3 = 0.0
        lambda4 = 0.0
    elif int(rodrope_block) == 3:
        m3 = 0.0
        lambda4 = 0.0
    return torch.tensor(
        [
            encode_lambda2(lambda2),
            encode_lambda3(lambda3) if int(rodrope_block) >= 3 else 0.0,
            encode_lambda4(lambda4) if int(rodrope_block) == 4 else 0.0,
            encode_m1(m1),
            encode_m2(m2) if int(rodrope_block) >= 3 else 0.0,
            encode_m3(m3) if int(rodrope_block) == 4 else 0.0,
        ],
        dtype=torch.float64,
    )


def objective_value(y_raw: Tensor) -> Tensor:
    """Single objective: F = f_acquisition + f_reasoning."""
    return y_raw.detach().to(dtype=torch.float64).sum().view(1)


class PerDimensionCategoricalKernel(Kernel):
    """Product categorical kernel: prod_i 1[m_i=m_i'] + eta_i * 1[m_i!=m_i']."""

    has_lengthscale = False

    def __init__(self, num_dims: int, **kwargs):
        super().__init__(**kwargs)
        self.num_dims = int(num_dims)
        self.register_parameter(
            name="raw_eta",
            parameter=torch.nn.Parameter(torch.zeros(self.num_dims, dtype=torch.float64)),
        )
        self.register_constraint("raw_eta", Interval(1e-6, 1.0 - 1e-6))

    @property
    def eta(self) -> Tensor:
        return self.raw_eta_constraint.transform(self.raw_eta)

    @eta.setter
    def eta(self, value: Tensor) -> None:
        self._set_eta(value)

    def _set_eta(self, value: Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.raw_eta.dtype, device=self.raw_eta.device)
        self.initialize(raw_eta=self.raw_eta_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        if x1.shape[-1] != self.num_dims or x2.shape[-1] != self.num_dims:
            raise ValueError(f"Expected {self.num_dims} categorical dimensions.")
        if diag:
            return torch.ones(*x1.shape[:-1], dtype=x1.dtype, device=x1.device)

        matches = torch.isclose(x1.unsqueeze(-2), x2.unsqueeze(-3), atol=1e-8, rtol=0.0)
        eta = self.eta.to(dtype=x1.dtype, device=x1.device)
        eta = eta.view(*([1] * (matches.dim() - 1)), self.num_dims)
        per_dim = torch.where(matches, torch.ones_like(matches, dtype=x1.dtype), eta)
        return per_dim.prod(dim=-1)


def kernel_dims_for_block(rodrope_block: int) -> Tuple[List[int], List[int]]:
    if int(rodrope_block) == 2:
        return [0], [3]
    if int(rodrope_block) == 3:
        return [0, 1], [3, 4]
    if int(rodrope_block) == 4:
        return [0, 1, 2], [3, 4, 5]
    raise ValueError("rodrope_block must be 2, 3 or 4.")


def build_product_kernel(rodrope_block: int) -> ScaleKernel:
    cont_dims, disc_dims = kernel_dims_for_block(rodrope_block)
    cont_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=len(cont_dims),
        active_dims=cont_dims,
    )
    disc_kernel = PerDimensionCategoricalKernel(
        num_dims=len(disc_dims),
        active_dims=disc_dims,
    )
    return ScaleKernel(cont_kernel * disc_kernel)


def build_model(train_X: Tensor, train_F: Tensor, rodrope_block: int) -> SingleTaskGP:
    train_Yvar = torch.full_like(train_F, GP_NOISE_VARIANCE)
    gp = SingleTaskGP(
        train_X,
        train_F,
        train_Yvar=train_Yvar,
        covar_module=build_product_kernel(rodrope_block),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def best_observed_index(train_F: Tensor) -> int:
    return int(train_F.view(-1).argmax().item())


def representative_norms(train_X: Tensor, train_F: Tensor) -> Tuple[int, float, float, float]:
    best_idx = best_observed_index(train_F)
    return (
        best_idx,
        float(train_X[best_idx, 0].item()),
        float(train_X[best_idx, 1].item()),
        float(train_X[best_idx, 2].item()),
    )


def evaluate_config(
    rodrope_block: int,
    m1: float,
    lambda2: float,
    m2: float = 0.0,
    lambda3: float = 0.0,
    m3: float = 0.0,
    lambda4: float = 0.0,
    run_tag: str = "manual",
    language: str = LANGUAGE,
    m: int = M,
    n: int = N,
    max_context_length: int = MAX_CONTEXT_LENGTH,
    model_path: str = MODEL_PATH,
    results_root: str = RESULTS_ROOT,
    counting_context_root: str = COUNTING_CONTEXT_ROOT,
    flash_impl: str = RODROPE_FLASH_IMPL,
) -> Tensor:
    rodrope_block = int(rodrope_block)
    if rodrope_block == 2:
        m2 = 0.0
        lambda3 = 0.0
        m3 = 0.0
        lambda4 = 0.0
    elif rodrope_block == 3:
        m3 = 0.0
        lambda4 = 0.0

    print(
        f"\n[eval_config] run_tag={run_tag}, block={rodrope_block}, "
        f"m1={m1}, lambda2={lambda2}, "
        f"m2={m2}, lambda3={lambda3}, "
        f"m3={m3}, lambda4={lambda4}",
        flush=True,
    )

    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable,
        script_path,
        "--worker",
        "--rodrope_block", str(rodrope_block),
        "--m1", str(int(m1)),
        "--lambda2", str(lambda2),
        "--m2", str(int(m2)),
        "--lambda3", str(lambda3),
        "--m3", str(int(m3)),
        "--lambda4", str(lambda4),
        "--language", str(language),
        "--m", str(int(m)),
        "--n", str(int(n)),
        "--max_context_length", str(int(max_context_length)),
        "--model_path", str(model_path),
        "--results_root", str(results_root),
        "--counting_context_root", str(counting_context_root),
        "--rodrope_flash_impl", str(flash_impl),
        "--run_tag", str(run_tag),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print("[eval_config] worker subprocess failed", flush=True)
        print("[eval_config] command:", " ".join(cmd), flush=True)
        if exc.stdout:
            print("[eval_config] worker stdout:\n" + exc.stdout, flush=True)
        if exc.stderr:
            print("[eval_config] worker stderr:\n" + exc.stderr, flush=True)
        raise

    data = parse_worker_result(result.stdout)
    f_acq = float(data["f_acq"])
    f_reason = float(data["f_reason"])
    print(
        f"[eval_config] acquisition={f_acq:.4f}, reasoning={f_reason:.4f}",
        flush=True,
    )

    return torch.tensor([f_acq, f_reason], dtype=torch.float64)


def parse_worker_result(stdout: str) -> dict:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "f_acq" in data and "f_reason" in data:
            return data
    raise RuntimeError(f"Worker did not emit a valid score JSON. stdout:\n{stdout}")


def random_initial_config(rodrope_block: int):
    m1, m2, m3 = random.choice(valid_discrete_configs_for_block(rodrope_block))
    lambda2 = random.uniform(LAMBDA2_MIN, LAMBDA2_MAX)
    lambda3 = random.uniform(LAMBDA3_MIN, LAMBDA3_MAX) if rodrope_block >= 3 else 0.0
    lambda4 = random.uniform(LAMBDA4_MIN, LAMBDA4_MAX) if rodrope_block == 4 else 0.0
    return m1, lambda2, m2, lambda3, m3, lambda4


def alternating_bo_for_block(
    rodrope_block: int,
    T: int = INNER_ITERATIONS,
    N0: int = INITIAL_SAMPLE_SIZE,
    raw_samples: int = 64,
    num_restarts: int = 5,
    maxiter: int = 50,
    seed: int = 0,
    xi: float = EI_XI,
    language: str = LANGUAGE,
    m: int = M,
    n: int = N,
    max_context_length: int = MAX_CONTEXT_LENGTH,
    model_path: str = MODEL_PATH,
    results_root: str = RESULTS_ROOT,
    counting_context_root: str = COUNTING_CONTEXT_ROOT,
    flash_impl: str = RODROPE_FLASH_IMPL,
) -> Tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(seed)
    random.seed(seed)
    rodrope_block = int(rodrope_block)

    train_X_list = []
    train_F_list = []
    train_raw_Y_list = []

    for i in range(N0):
        m1, lambda2, m2, lambda3, m3, lambda4 = random_initial_config(rodrope_block)

        t0 = time.time()
        y_raw = evaluate_config(
            rodrope_block=rodrope_block,
            m1=m1,
            lambda2=lambda2,
            m2=m2,
            lambda3=lambda3,
            m3=m3,
            lambda4=lambda4,
            run_tag=f"block{rodrope_block}_init{i}",
            language=language,
            m=m,
            n=n,
            max_context_length=max_context_length,
            model_path=model_path,
            results_root=results_root,
            counting_context_root=counting_context_root,
            flash_impl=flash_impl,
        )
        f_value = objective_value(y_raw)
        elapsed = time.time() - t0

        train_X_list.append(
            encode_config(
                rodrope_block=rodrope_block,
                m1=m1,
                lambda2=lambda2,
                m2=m2,
                lambda3=lambda3,
                m3=m3,
                lambda4=lambda4,
            )
        )
        train_F_list.append(f_value)
        train_raw_Y_list.append(y_raw)

        print(
            f"[init][block={rodrope_block}] i={i}, m1={m1}, lambda2={lambda2:.3f}, "
            f"m2={m2}, lambda3={lambda3:.3f}, "
            f"m3={m3}, lambda4={lambda4:.3f}, "
            f"raw={y_raw.tolist()}, F={float(f_value.item()):.4f}, time={elapsed:.1f}s"
        )

        log_eval_result(
            stage="init",
            outer_iter=-1,
            init_index=i,
            rodrope_block=rodrope_block,
            m1=m1,
            lambda2=lambda2,
            m2=m2,
            lambda3=lambda3,
            m3=m3,
            lambda4=lambda4,
            f_acq=float(y_raw[0].item()),
            f_reason=float(y_raw[1].item()),
            elapsed_sec=elapsed,
        )

    train_X = torch.stack(train_X_list, dim=0)
    train_F = torch.stack(train_F_list, dim=0).view(-1, 1)
    train_raw_Y = torch.stack(train_raw_Y_list, dim=0)

    best_idx, lambda2_t_norm, lambda3_t_norm, lambda4_t_norm = representative_norms(train_X, train_F)
    print(
        f"[init][block={rodrope_block}] best index = {best_idx}, "
        f"best_F = {float(train_F[best_idx].item()):.4f}, "
        f"lambda2_0_norm = {lambda2_t_norm:.4f}, lambda3_0_norm = {lambda3_t_norm:.4f}, "
        f"lambda4_0_norm = {lambda4_t_norm:.4f}"
    )

    bounds = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )

    for t in range(T):
        print(f"\n=== Fixed-block BO Iter {t} | block={rodrope_block} ===")
        model = build_model(train_X, train_F, rodrope_block)
        best_f = float(train_F.max().item())
        acqf = ExpectedImprovement(model=model, best_f=best_f + xi, maximize=True)

        best_candidate = None
        best_acq_value = -float("inf")
        best_cfg = None
        for m1, m2, m3 in valid_discrete_configs_for_block(rodrope_block):
            m1_norm = encode_m1(m1)
            m2_norm = encode_m2(m2) if rodrope_block >= 3 else 0.0
            m3_norm = encode_m3(m3) if rodrope_block == 4 else 0.0
            fixed_features = {
                3: m1_norm,
                4: m2_norm,
                5: m3_norm,
            }
            if rodrope_block == 2:
                fixed_features.update({1: 0.0, 2: 0.0})
            elif rodrope_block == 3:
                fixed_features.update({2: 0.0})

            candidate, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={"batch_limit": 5, "maxiter": maxiter},
                fixed_features=fixed_features,
            )
            acq_scalar = float(acq_value.item())
            if acq_scalar > best_acq_value:
                best_acq_value = acq_scalar
                best_candidate = candidate.detach()
                best_cfg = (m1, m2, m3)

        if best_candidate is None or best_cfg is None:
            raise RuntimeError(f"Failed to select EI candidate for block={rodrope_block}.")

        m1_t1, m2_t1, m3_t1 = best_cfg
        lambda2_t1_norm = float(best_candidate[0, 0].item())
        lambda3_t1_norm = float(best_candidate[0, 1].item()) if rodrope_block >= 3 else 0.0
        lambda4_t1_norm = float(best_candidate[0, 2].item()) if rodrope_block == 4 else 0.0
        lambda2_t1 = decode_lambda2(lambda2_t1_norm)
        lambda3_t1 = decode_lambda3(lambda3_t1_norm) if rodrope_block >= 3 else 0.0
        lambda4_t1 = decode_lambda4(lambda4_t1_norm) if rodrope_block == 4 else 0.0
        print(
            f"[EI][block={rodrope_block}] chosen m1={m1_t1}, m2={m2_t1}, m3={m3_t1}, "
            f"lambda2={lambda2_t1:.4f} (norm={lambda2_t1_norm:.4f}), "
            f"lambda3={lambda3_t1:.4f} (norm={lambda3_t1_norm:.4f}), "
            f"lambda4={lambda4_t1:.4f} (norm={lambda4_t1_norm:.4f}), "
            f"EI={best_acq_value:.6f}, best_F={best_f:.4f}, xi={xi}"
        )

        t0 = time.time()
        f_raw_t1 = evaluate_config(
            rodrope_block=rodrope_block,
            m1=m1_t1,
            lambda2=lambda2_t1,
            m2=m2_t1,
            lambda3=lambda3_t1,
            m3=m3_t1,
            lambda4=lambda4_t1,
            run_tag=f"block{rodrope_block}_BO{t + 1}",
            language=language,
            m=m,
            n=n,
            max_context_length=max_context_length,
            model_path=model_path,
            results_root=results_root,
            counting_context_root=counting_context_root,
            flash_impl=flash_impl,
        )
        f_value_t1 = objective_value(f_raw_t1)
        elapsed = time.time() - t0

        print(
            f"[eval][block={rodrope_block}] m1={m1_t1}, lambda2={lambda2_t1:.4f}, "
            f"m2={m2_t1}, lambda3={lambda3_t1:.4f}, "
            f"m3={m3_t1}, lambda4={lambda4_t1:.4f}, "
            f"raw={f_raw_t1.tolist()}, F={float(f_value_t1.item()):.4f}, time={elapsed:.1f}s"
        )

        log_eval_result(
            stage="bo_iter",
            outer_iter=t,
            init_index=-1,
            rodrope_block=rodrope_block,
            m1=m1_t1,
            lambda2=lambda2_t1,
            m2=m2_t1,
            lambda3=lambda3_t1,
            m3=m3_t1,
            lambda4=lambda4_t1,
            f_acq=float(f_raw_t1[0].item()),
            f_reason=float(f_raw_t1[1].item()),
            elapsed_sec=elapsed,
        )

        new_X = encode_config(
            rodrope_block=rodrope_block,
            m1=m1_t1,
            lambda2=lambda2_t1,
            m2=m2_t1,
            lambda3=lambda3_t1,
            m3=m3_t1,
            lambda4=lambda4_t1,
        ).unsqueeze(0)
        train_X = torch.cat([train_X, new_X], dim=0)
        train_F = torch.cat([train_F, f_value_t1.view(1, 1)], dim=0)
        train_raw_Y = torch.cat([train_raw_Y, f_raw_t1.unsqueeze(0)], dim=0)

        best_idx, lambda2_t_norm, lambda3_t_norm, lambda4_t_norm = representative_norms(train_X, train_F)
        print(
            f"[update][block={rodrope_block}] best idx={best_idx}, "
            f"best_F={float(train_F[best_idx].item()):.4f}, "
            f"lambda2_repr_norm={lambda2_t_norm:.4f}, "
            f"lambda3_repr_norm={lambda3_t_norm:.4f}, lambda4_repr_norm={lambda4_t_norm:.4f}"
        )

    best_idx = best_observed_index(train_F)
    best_X = train_X[best_idx]
    best_raw_Y = train_raw_Y[best_idx]
    best_F = train_F[best_idx]
    lambda2_norm, lambda3_norm, lambda4_norm, m1_norm, m2_norm, m3_norm = best_X.tolist()
    lambda2 = decode_lambda2(lambda2_norm)
    m1 = decode_m1(m1_norm)
    if rodrope_block >= 3:
        m2 = decode_m2(m2_norm)
        lambda3 = decode_lambda3(lambda3_norm)
    else:
        m2 = 0.0
        lambda3 = 0.0
    if rodrope_block == 4:
        m3 = decode_m3(m3_norm)
        lambda4 = decode_lambda4(lambda4_norm)
    else:
        m3 = 0.0
        lambda4 = 0.0
    f_acq, f_reason = best_raw_Y.tolist()
    print(f"\n===== Best Observed Solution | block={rodrope_block} =====")
    print(
        f"  block={rodrope_block}, m1={m1}, lambda2={lambda2:.3f}, "
        f"m2={m2}, lambda3={lambda3:.3f}, m3={m3}, lambda4={lambda4:.3f} "
        f"-> acq={f_acq:.4f}, reason={f_reason:.4f}, F={float(best_F.item()):.4f}"
    )

    return best_X, best_raw_Y, best_F

def alternating_bo(
    T: int = INNER_ITERATIONS,
    N0: int = INITIAL_SAMPLE_SIZE,
    raw_samples: int = 64,
    num_restarts: int = 5,
    maxiter: int = 50,
    seed: int = 0,
    blocks: Tuple[int, ...] = (2, 3, 4),
    xi: float = EI_XI,
    language: str = LANGUAGE,
    m: int = M,
    n: int = N,
    max_context_length: int = MAX_CONTEXT_LENGTH,
    model_path: str = MODEL_PATH,
    results_root: str = RESULTS_ROOT,
    counting_context_root: str = COUNTING_CONTEXT_ROOT,
    flash_impl: str = RODROPE_FLASH_IMPL,
):
    set_experiment_dir(results_root)
    init_log_file()
    results = {}
    for offset, rodrope_block in enumerate(blocks):
        print(f"\n######## Start fixed block={rodrope_block} optimization ########")
        results[int(rodrope_block)] = alternating_bo_for_block(
            rodrope_block=int(rodrope_block),
            T=T,
            N0=N0,
            raw_samples=raw_samples,
            num_restarts=num_restarts,
            maxiter=maxiter,
            seed=seed + offset,
            xi=xi,
            language=language,
            m=m,
            n=n,
            max_context_length=max_context_length,
            model_path=model_path,
            results_root=results_root,
            counting_context_root=counting_context_root,
            flash_impl=flash_impl,
        )
    return results

# =====================================================================
# 6. 主入口
# =====================================================================

def parse_blocks_arg(blocks_arg: str) -> Tuple[int, ...]:
    blocks = tuple(int(part.strip()) for part in blocks_arg.split(",") if part.strip())
    if not blocks or any(block not in (2, 3, 4) for block in blocks):
        raise ValueError("--blocks must be a comma-separated subset of 2,3,4.")
    return blocks


def parse_main_args():
    parser = argparse.ArgumentParser(description="Fixed-block BO for Llama-3 Rodrope Counting-Stars objectives.")
    parser.add_argument("--T", type=int, default=INNER_ITERATIONS)
    parser.add_argument("--N0", type=int, default=INITIAL_SAMPLE_SIZE)
    parser.add_argument("--raw_samples", type=int, default=256)
    parser.add_argument("--num_restarts", type=int, default=256)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--blocks", type=str, default="2,3,4")
    parser.add_argument("--xi", type=float, default=EI_XI)
    parser.add_argument("--language", type=str, default=LANGUAGE, choices=["EN", "ZH"])
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--max_context_length", type=int, default=MAX_CONTEXT_LENGTH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--results_root", type=str, default=RESULTS_ROOT)
    parser.add_argument("--counting_context_root", type=str, default=COUNTING_CONTEXT_ROOT)
    parser.add_argument("--rodrope_flash_impl", type=str, default=RODROPE_FLASH_IMPL, choices=["flash_attn", "triton"])
    return parser.parse_args()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_main()
    else:
        args = parse_main_args()
        best_results = alternating_bo(
            T=args.T,
            N0=args.N0,
            raw_samples=args.raw_samples,
            num_restarts=args.num_restarts,
            maxiter=args.maxiter,
            seed=args.seed,
            blocks=parse_blocks_arg(args.blocks),
            xi=args.xi,
            language=args.language,
            m=args.m,
            n=args.n,
            max_context_length=args.max_context_length,
            model_path=args.model_path,
            results_root=args.results_root,
            counting_context_root=args.counting_context_root,
            flash_impl=args.rodrope_flash_impl,
        )

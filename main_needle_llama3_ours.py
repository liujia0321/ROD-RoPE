"""
Needle-in-a-Haystack test with LLaMA-3 + configurable fixed-block Rodrope(flash_attn).


import tiktoken
import os
import glob
import json
# import tensor_parallel as tp  # 不再使用 tensor_parallel，可删
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# from anthropic import Anthropic
# from dotenv import load_dotenv
import numpy as np
import argparse
from rouge_score import rouge_scorer
# import tensor_parallel as tp
from transformers import pipeline

import random
import numpy as np
import torch

# ===== Rodrope 相关 =====
import warnings
warnings.filterwarnings("ignore")

import Rodrope  # 使用你 passkey/Counting-Stars 
from torch import nn

from openai import OpenAI
from datetime import datetime, timezone
import time


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

# ============= 原脚本的全局对象 =============
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="/home/Rodrope/context_data/PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 results_version = 1,
                 context_lengths_min = 2048,
                 context_lengths_max = 32768,
                 context_lengths_num_intervals = 31,
                 context_lengths = None,
                 document_depth_percent_min = 10,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 rodrope_block = 3,
                 m1 = 256,
                 lambda2 = 6.85,
                 m2 = 24576,
                 lambda3 = 8.0,
                 m3 = None,
                 lambda4 = None,
                 rodrope_enable_flash_attention = True,
                 rodrope_flash_impl = "flash_attn",
                 rodrope_scale_base = -1):
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []

        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else:
            self.model_version = model_name
        if(model_name_suffix is not None):
            self.model_version += "_" + model_name_suffix
        if rodrope_block == 2:
            self.model_version += f"_2block_m1{m1}_lambda2{lambda2}"
        elif rodrope_block == 3 and m2 is not None and lambda3 is not None:
            self.model_version += f"_3block_m1{m1}_lambda2{lambda2}_m2{m2}_lambda3{lambda3}"
        elif rodrope_block == 4 and m2 is not None and lambda3 is not None and m3 is not None and lambda4 is not None:
            self.model_version += f"_4block_m1{m1}_lambda2{lambda2}_m2{m2}_lambda3{lambda3}_m3{m3}_lambda4{lambda4}"

        # ====== context 长度网格 ======
        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(
                    np.linspace(context_lengths_min, context_lengths_max,
                                num=context_lengths_num_intervals, endpoint=True)
                ).astype(int)
        else:
            self.context_lengths = context_lengths

        # ====== 深度百分比网格 ======
        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(
                        np.linspace(document_depth_percent_min, document_depth_percent_max,
                                    num=document_depth_percent_intervals, endpoint=True)
                    ).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'.")

        self.model_name = model_name

        # ====== 模型加载部分 ======
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            # 对 LLaMA/Mistral/GLM 等本地模型走这里
            print("loading from %s" % model_name)

            # tokenizer：用 HF 的 AutoTokenizer
            self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.enc.pad_token is None:
                self.enc.pad_token = self.enc.eos_token
            self.enc.padding_side = "left"

            cfg = AutoConfig.from_pretrained(model_name)
            # 如果以后用 Mistral 可以在这里关闭 sliding_window
            if getattr(cfg, "model_type", "") == "mistral":
                cfg.sliding_window = None

            # 多卡：device_map="auto"，并启用 flash_attention_2
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            try:
                self.model_to_test = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=cfg,
                    torch_dtype=dtype,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
            except Exception as e:
                print(f"[warn] flash_attention_2 load failed: {e}\n       fallback to eager.")
                self.model_to_test = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=cfg,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                    device_map="auto",
                )

            # RoPE 适配 + 可配置 fixed-block Rodrope.apply（flash_attn 模式）
            patch_rope_for_rodrope(self.model_to_test)
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
            Rodrope.apply(self.model_to_test, **apply_kwargs)
            print(
                f"[Rodrope] applied: block={rodrope_block}, m1={m1}, lambda2={lambda2}, "
                f"m2={m2}, lambda3={lambda3}, m3={m3}, lambda4={lambda4}, "
                f"enable_flash_attention={rodrope_enable_flash_attention}, flash_impl={rodrope_flash_impl}, "
                f"scale_base={rodrope_scale_base}"
            )

            self.model_to_test.eval()
        else:
            # OpenAI / Anthropic 分支，保持原来的逻辑
            self.model_to_test = OpenAI(api_key=openai_api_key)
            if(self.model_provider == "OpenAI"):
                self.enc = tiktoken.encoding_for_model(self.model_name)
            elif(self.model_provider == "Anthropic"):
                # self.enc = Anthropic().get_tokenizer()
                raise NotImplementedError("Anthropic branch tokenizer not implemented in this script.")

        self.model_to_test_description = model_name
        
        self.evaluation_model = None
        self.debug = 'debug'
        model_name = model_name.split('/')[-1]

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len:
                continue
            for depth_percent in self.document_depth_percents:
                self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            # 本地模型：走 LLaMA 风格 prompt
            test_format = (
                f"<|im_start|> This is a very long story book: <book> {context} </book>.\n"
                f" Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            )
            return test_format
        else:
            # OpenAI/Anthropic 分支
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",  # 这里留空
                },
            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # 已有结果则跳过
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        # 构造上下文 + 插入 needle
        context = self.generate_context(context_length, depth_percent)

        # 准备 prompt
        prompt = self.generate_prompt(context)
        test_start_time = time.time()

        if(self.model_provider in ["OpenAI", "Anthropic"]):
            # OpenAI/Anthropic -> 用 API
            response = self.model_to_test.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=300,
                temperature=0
            )
            response = response.choices[0].message.content
        else:
            
            enc = self.enc(prompt, return_tensors="pt")
            input_ids = enc["input_ids"]          # 不手动 .to(device)
            attention_mask = enc["attention_mask"]

            with torch.no_grad():
                output_ids = self.model_to_test.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.enc.pad_token_id,
                    eos_token_id=self.enc.eos_token_id,
                )
                response = self.enc.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        score = scorer.score(self.needle, response)['rouge1'].fmeasure * 10

        results = {
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] = context_file_location

            if not os.path.exists('contexts'):
                os.makedirs('contexts')
            if not os.path.exists(f'contexts/{self.model_version}'):
                os.makedirs(f'contexts/{self.model_version}')

            with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            if not os.path.exists('results'):
                os.makedirs('results')
            
            if not os.path.exists(f'results/{self.model_version}'):
                os.makedirs(f'results/{self.model_version}')

            p = f'results/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    # ====== 下面这部分是原脚本的 context 构造与编码 ======
    def generate_context(self, context_length, depth_percent):
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needle(context, depth_percent, context_length)
        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        context_length -= self.final_context_length_buffer

        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]

            if(self.model_provider in ["LLaMA", "LongLLaMA"]):
                period_tokens = [29889, 869]
            elif(self.model_provider == "Mistral"):
                period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"):
                period_tokens = [918, 30930]
            else:
                period_tokens = self.encode_text_to_tokens('.')
            
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            encoded = self.enc.encode(context)
            return len(encoded.ids)
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, default=0, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, default=160000, help='a number')
    parser.add_argument('--model_path', type=str, default='/path/to/meta-llamaMeta-Llama-3-8B-Instruct', help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--rodrope_block', type=int, default=2, choices=[2, 3, 4], help='2/3/4-block Rodrope')
    parser.add_argument('--m1', type=int, default=2048, help='neighbor window size')
    parser.add_argument('--lambda2', type=float, default=16.0, help='compression ratio for the first grouped block')
    parser.add_argument('--m2', type=int, default=4096, help='distance boundary where the far block starts')
    parser.add_argument('--lambda3', type=float, default=16.0, help='compression ratio for the far block')
    parser.add_argument('--m3', type=int, default=None, help='distance boundary where the far2 block starts')
    parser.add_argument('--lambda4', type=float, default=None, help='compression ratio for the far2 block')
    parser.add_argument('--rodrope_flash_impl', type=str, default="flash_attn", choices=["flash_attn", "triton"], help='fixed-block path supports flash_attn')
    parser.add_argument('--rodrope_scale_base', type=int, default=-1, help='query scaling base; -1 disables scaling')
    args = parser.parse_args()

    if(args.model_path is not None):
        assert(args.model_name is None)
        model_name = args.model_path
    else:
        assert(args.model_name is not None)
        model_name = args.model_name

    # 直接在这里写要测试的 Rodrope 配置；脚本会依次测试每一组。
    # block=2 时 m2/lambda3/m3/lambda4 写 None；block=3 时 m3/lambda4 写 None。
    NEEDLE_SE_CONFIGS = [
        {
            "rodrope_block": 2,
            "m1": 2048,
            "lambda2": 16.0,
            "m2": None,
            "lambda3": None,
            "m3": None,
            "lambda4": None,
        },
        # {
        #     "rodrope_block": 3,
        #     "m1": 256,
        #     "lambda2": 6.85,
        #     "m2": 24576,
        #     "lambda3": 8.0,
        #     "m3": None,
        #     "lambda4": None,
        # },
    ]

    if args.rodrope_flash_impl == "triton" and any(cfg["rodrope_block"] >= 3 for cfg in NEEDLE_SE_CONFIGS):
        raise NotImplementedError("block>=3 is implemented for --rodrope_flash_impl flash_attn.")

    for config_index, rodrope_config in enumerate(NEEDLE_SE_CONFIGS, start=1):
        print()
        print(
            f"[main] testing config {config_index}/{len(NEEDLE_SE_CONFIGS)}: "
            f"block={rodrope_config['rodrope_block']}, m1={rodrope_config['m1']}, "
            f"lambda2={rodrope_config['lambda2']}, m2={rodrope_config['m2']}, "
            f"lambda3={rodrope_config['lambda3']}, m3={rodrope_config['m3']}, "
            f"lambda4={rodrope_config['lambda4']}"
        )

        ht = LLMNeedleHaystackTester(model_name=model_name,
                                     model_name_suffix=args.model_name_suffix,
                                     model_provider=args.model_provider,
                                     save_contexts=True,
                                     save_results=True,
                                     openai_api_key=args.api_key,
                                     rodrope_block=rodrope_config["rodrope_block"],
                                     m1=rodrope_config["m1"],
                                     lambda2=rodrope_config["lambda2"],
                                     m2=rodrope_config["m2"],
                                     lambda3=rodrope_config["lambda3"],
                                     m3=rodrope_config["m3"],
                                     lambda4=rodrope_config["lambda4"],
                                     rodrope_flash_impl=args.rodrope_flash_impl,
                                     rodrope_scale_base=args.rodrope_scale_base
                                     )

        ht.start_test(args)

        del ht
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

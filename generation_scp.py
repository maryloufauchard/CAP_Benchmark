import os
import re
import sys
import time
import random
import copy
import argparse
import importlib.util
from os import walk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI

import torch
torch.cuda.empty_cache()


# find a similar example of a smaller instance for ICL
def find_example_files(instance_filename):
    pattern = r"scp_\((\d+),(\d+)\)_(\w+)_(\d+)_seed(\d+)\.txt"
    match = re.match(pattern, instance_filename)

    if not match:
        raise ValueError(f"Filename {instance_filename} doesn't match expected pattern.")
    n_students, n_schools, pref_mode, total_cap, seed = match.groups()
    n_students = int(n_students)
    n_schools = int(n_schools)
    total_cap = int(total_cap)
    ratio = n_students / n_schools
    ideal_n_schools = min(5 if abs(ratio - 1.0) <= 0.3 else 3, 4)
    capacity_ratio = total_cap / n_students
    if capacity_ratio <= 0.85:
        example_total_cap = 4
    elif capacity_ratio <= 1.05:
        example_total_cap = 5
    else:
        example_total_cap = 10

    example_n_schools = min(example_total_cap, ideal_n_schools)
    example_seed = random.choice([10, 20, 30])

    base_name = f"scp_(5,{example_n_schools})_{pref_mode}_{example_total_cap}_seed{example_seed}"
    example_filename = f"ex_{base_name}.txt"
    match_filename = f"ex_match_{base_name}.txt"

    return example_filename, match_filename


def extract_final_channel(text: str) -> str:
    marker = "<|channel|>final<|message|>"
    if marker not in text:
        return text
    seg = text.split(marker, 1)[1]
    stops = ("<|end|>", "<|eot_id|>", "<|return|>", "<|stop|>")
    idxs = [seg.find(s) for s in stops if s in seg]
    if idxs:
        return seg[:min(idxs)].strip()
    return seg.strip()


def generate_with_qwq_vllm(model, tokenizer, prompt_text, base_sampling_params, strip_reasoning=True, stop_words=None):
    messages = [
        {"role": "user", "content": prompt_text},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    sp = copy.deepcopy(base_sampling_params)
    if stop_words is not None:
        sp.stop = stop_words
    if getattr(sp, "max_tokens", None) is None or sp.max_tokens < 32768:
        sp.max_tokens = 32768
    outs = model.generate([chat_text], sampling_params=sp, use_tqdm=False)
    out = outs[0].outputs[0]
    text = out.text or ""
    if strip_reasoning:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if getattr(out, "finish_reason", None) == "length" and sp.max_tokens < 65536:
        sp = SamplingParams(
            max_tokens=min(65536, sp.max_tokens * 2),
            temperature=sp.temperature, top_p=sp.top_p, top_k=sp.top_k,
            repetition_penalty=getattr(sp, "repetition_penalty", 1.0),
            stop=getattr(sp, "stop", None)
        )
        out = model.generate([chat_text], sampling_params=sp, use_tqdm=False)[0].outputs[0]
        text = out.text or ""
        if strip_reasoning:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def generate_with_openai(prompt, model_name, params, client):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=32000
    )
    return response.choices[0].message.content


def generate_with_vllm_server(prompt, server_client, server_model_id, sampling_params):
    response = server_client.chat.completions.create(
        model=server_model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        extra_body={
            "top_k": sampling_params.top_k,
            "repetition_penalty": getattr(sampling_params, "repetition_penalty", 1.0),
        },
    )
    return response.choices[0].message.content


current_path = os.getcwd()
home_path = base_path = os.path.join(*current_path.split(os.sep)[:5])


#############################################################################################
##############################   PARSER ARGUMENTS   #########################################
parser = argparse.ArgumentParser(description="LLM Generation of many-to-one matching problem")
parser.add_argument("--model", default='llama3-big',
                     help="LLMs model for generation. Expected values : llama3-big | llama3-instruct | "
                          "mistral-instruct | qwen2-7b-instruct | qwq-32b | gpt-oss-120b | gpt-5-mini | o3")
parser.add_argument("--run", default=0, type=int, help="indicate the run number")
parser.add_argument("--config", default=1, type=int,
                     help="indicate the config number to use -- 3 possibilities 1 for balanced crea 2 for strict model 3 for divergent generation")
parser.add_argument("--cache_dir", default=None, help="path to model cache directory")
parser.add_argument("--save_path", default=None, help="path to general directory")
parser.add_argument('--model_token_len', default=8192, type=int, help="input token length for vLLM models")
parser.add_argument(
    "--config_file",
    required=True,
    help="path to the config .py you want to use (e.g. /path/to/my_config.py)"
)
parser.add_argument('--nb_gpus', default=2, type=int, help='tensor_parallel_size in vLLM for large models only')

args = parser.parse_args()

SERVER_MODELS = ['gpt-oss-120b']
SERVER_URL = "http://localhost:8000/v1"

use_server = args.model in SERVER_MODELS

config_path = os.path.abspath(args.config_file)
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Could not find config file at {config_path}")

spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

args.cache_dir = args.cache_dir or config.path["cache_dir"]
args.save_path = args.save_path or config.path["save_dir"]


if args.model == 'llama3-big':
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
elif args.model == 'llama3-instruct':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
elif args.model == 'qwen2-7b-instruct':
    model_name = "Qwen/Qwen2-7B-Instruct"
elif args.model == 'qwq-32b':
    model_name = 'Qwen/QwQ-32B'
elif args.model == 'gpt-oss-120b':
    model_name = 'openai/gpt-oss-120b'
elif args.model == 'mistral-instruct':
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
elif args.model == "gpt-5-mini":
    model_name = "gpt-5-mini"
elif args.model == "o3":
    model_name = "o3"
else:
    print('Please provide a valid entry for models -- ')
    raise KeyError


openai_model = ['gpt-4o', 'gpt-4o-mini', 'gpt-5-mini', 'o3', 'o4-mini']
big_model_list = ['llama3-big', 'gemma-2-big', 'falcon-big', 'qwen2-7b-instruct',
                   'mistral-instruct', 'qwq-32b', 'gpt-oss-120b', 'gpt-5-mini']

openai_client = OpenAI() if args.model in openai_model else None

tokenizer = None
model = None
server_client = None
server_model_id = None

if use_server:
    server_client = OpenAI(api_key="EMPTY", base_url=SERVER_URL)
    available_models = [m.id for m in server_client.models.list().data]
    server_model_id = available_models[0]

elif args.model not in openai_model:

    if args.model in ['qwen2-7b-instruct', 'mistral-instruct', 'qwq-32b', 'gpt-oss-120b']:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

    max_len = args.model_token_len
    if args.model in ['qwq-32b', 'gpt-oss-120b']:
        max_len = max(max_len, 40960)

    if args.model in big_model_list:
        dtype_m = 'float16'
        if args.model in ['gemma-2-big']:
            dtype_m = "bfloat16"
        print('BIG MODEL')
        model = LLM(model=model_name, revision='main', max_model_len=max_len, max_num_batched_tokens=max_len,
                    tokenizer=model_name, download_dir=args.cache_dir, trust_remote_code=True,
                    dtype=dtype_m, tensor_parallel_size=args.nb_gpus)
    else:
        model = LLM(model=model_name, revision='main', max_model_len=max_len, max_num_batched_tokens=max_len,
                    tokenizer=model_name, download_dir=args.cache_dir, trust_remote_code=True, dtype='float16')

else:
    # openai
    pass

configs = config.hp_balanced
config_ = f'config_{args.config}'
run = f'run_{args.run}'

model_configs = args.model + run + config_
save_path = args.save_path + f'Generated_answers_{model_configs}/'

save_directory = os.path.dirname(save_path)
if save_directory and not os.path.exists(save_directory):
    os.makedirs(save_directory)
    already_saved_files = []
    print(f"Created directory: {save_directory}")
else:
    already_saved_files = list(set(next(walk(save_path), (None, None, []))[2]))


tradi_gen_len = 10000

if use_server:
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=configs['temp_value'],
        top_p=configs['top_p_value'],
        top_k=configs['top_k_value'],
        repetition_penalty=configs['repet_value'],
    )
elif args.model in ['qwq-32b', 'gpt-oss-120b']:
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=configs['temp_value'],
        top_p=configs['top_p_value'],
        top_k=configs['top_k_value'],
        repetition_penalty=configs['repet_value'],
    )
elif args.model in openai_model:
    sampling_params = SamplingParams(
        max_tokens=10000,
        temperature=configs['temp_value'],
        top_p=configs['top_p_value'],
        top_k=configs['top_k_value'], # unused by OpenAI
        repetition_penalty=configs['repet_value'], # unused by OpenAI
    )
else:
    sampling_params = SamplingParams(
        max_tokens=tradi_gen_len,
        temperature=configs['temp_value'],
        top_p=configs['top_p_value'],
        top_k=configs['top_k_value'],
        repetition_penalty=configs['repet_value']
    )

instance_dir = config.path['instance_dir']
instruction_dir = config.path['instruction_dir']
generation_dir = config.path['output_path']
example_dir = config.path['example_dir']
example_match_dir = config.path['example_match_dir']

instruction_files = [f for f in os.listdir(instruction_dir) if f.endswith('.txt')]
instance_files = [f for f in os.listdir(instance_dir) if f.endswith('.txt')]

os.makedirs(generation_dir, exist_ok=True)


for instruction_file in tqdm(instruction_files, desc="Instructions"):
    instruction_path = os.path.join(instruction_dir, instruction_file)
    with open(instruction_path, 'r', encoding='utf-8') as f:
        instruction_text = f.read()

    instruction_name = os.path.splitext(instruction_file)[0]
    needs_example = "[ADD EXAMPLE HERE]" in instruction_text

    for instance_file in tqdm(instance_files, desc=f"Instances for {instruction_name}", leave=False):
        instance_base = os.path.splitext(instance_file)[0]
        output_filename = f"generation_{instruction_name}_{instance_base}.txt"
        output_path = os.path.join(generation_dir, output_filename)

        if os.path.exists(output_path):
            print(f" --> Skipping {output_filename}, already generated.")
            continue

        instance_path = os.path.join(instance_dir, instance_file)
        with open(instance_path, 'r', encoding='utf-8') as f:
            instance_text = f.read()

        prompt = instruction_text

        # ICL case
        if needs_example:
            try:
                ex_filename, match_filename = find_example_files(instance_file)
                ex_path = os.path.join(example_dir, ex_filename)
                match_path = os.path.join(example_match_dir, match_filename)

                with open(ex_path, 'r', encoding='utf-8') as f:
                    ex_content = f.read()
                with open(match_path, 'r', encoding='utf-8') as f:
                    match_content = f.read()

                example_section = f"If you have this instance:\n{ex_content}\n\nYou should return this matching in this format:\n{match_content}"
                prompt = prompt.replace("[ADD EXAMPLE HERE]", example_section)

            except FileNotFoundError:
                print(f"!!! Missing: {ex_filename} or {match_filename} for {instance_file}")
                prompt = prompt.replace("[ADD EXAMPLE HERE]", "[EXAMPLE MISSING]")

        prompt = prompt.replace("[ADD INSTANCE HERE]", instance_text)

        #
        start = time.time()
        if use_server:
            answer = generate_with_vllm_server(prompt, server_client, server_model_id, sampling_params)
        elif args.model in ['qwq-32b']:
            answer = generate_with_qwq_vllm(
                model, tokenizer, prompt, sampling_params,
                strip_reasoning=True,
                stop_words=None
            )
        elif args.model in openai_model:
            answer = generate_with_openai(prompt, model_name, sampling_params, openai_client)
        else:
            outputs = model.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
            answer = outputs[0].outputs[0].text
        duration = time.time() - start
        #

        answer = extract_final_channel(answer)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(answer)

        print(f"Done {output_filename} in {duration:.1f}s")

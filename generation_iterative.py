import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import statistics
import numpy as np
import json
import os
from os import walk
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import re
import importlib.util
import sys
torch.cuda.empty_cache()
import random
import ast
import copy
from collections import defaultdict


def extract_matching_from_text(text):
    if not text:
        return []

    text = (text.replace('\u2018', "'").replace('\u2019', "'")
                 .replace('\u201c', '"').replace('\u201d', '"'))
    text = re.sub(r"[ \t]+", " ", text)

    # Patterns
    block_pattern = r"\[\s*((?:\(\s*[\"']?s\d+[\"']?\s*,\s*[\"']?(?:c\d+|nothing)[\"']?\s*\)\s*,?\s*)+)\]"
    line_tuple_pattern = r"\(\s*[\"']?(s\d+)[\"']?\s*,\s*[\"']?(c\d+|nothing)[\"']?\s*\)\s*,?"
    json_array_pattern = r'\[\s*"?(s\d+)"?\s*,\s*"?(c\d+|nothing)"?\s*\](?:,|\s|$)'

    best_raw = None
    best_count = 0
    for block in re.finditer(block_pattern, text, re.DOTALL):
        raw = block.group(1)
        pairs = re.findall(line_tuple_pattern, raw)
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_raw = raw

    if best_raw:
        matches = re.findall(line_tuple_pattern, best_raw)
    else:
        lines = text.splitlines()
        collected = []
        started = False
        for ln in lines:
            if re.search(line_tuple_pattern, ln):
                collected.append(ln.strip())
                started = True
            elif started:
                break
        block_text = " ".join(collected)
        matches = re.findall(line_tuple_pattern, block_text)

    #Fallback to JSON-like pairs if needed
    if not matches:
        matches = re.findall(json_array_pattern, text)

    return [(s, c) for s, c in matches] if matches else []


def read_matching(file_path, is_list_format=False):
    matchings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    if not lines:
        return []
    if is_list_format:
        return ast.literal_eval(lines[0])
    for ln in lines:
        m = re.match(r"\(\s*['\"]?(s\d+)['\"]?\s*,\s*['\"]?(c\d+|nothing)['\"]?\s*\)", ln)
        if m:
            matchings.append((m.group(1), m.group(2)))
    return matchings


def parse_instance(file_path):
    capacities = {}
    student_prefs = {}
    college_priorities = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    while lines and lines[0].startswith('c') and ' ' in lines[0]:
        college, cap = lines.pop(0).split()
        capacities[college] = int(cap)
    while lines and lines[0].startswith('s'):
        parts = lines.pop(0).split()
        student = parts[0]
        prefs = [p.split(',')[1].rstrip(')') for p in parts[1:]]
        student_prefs[student] = prefs
    for ln in lines:
        parts = ln.split()
        college = parts[0]
        priorities = [p.split(',')[1].rstrip(')') for p in parts[1:]]
        college_priorities[college] = priorities
    return capacities, student_prefs, college_priorities


def compute_feasibility(matching, capacities):
    assigned = defaultdict(int)
    for _, col in matching:
        if col != 'nothing':
            assigned[col] += 1
    total = 0
    for c, cnt in assigned.items():
        cap = capacities.get(c, 0)
        if cnt > cap:
            total += cnt - cap
    return total


def compute_correct_pairs(llm_matching, real_matching):
    return len(set(llm_matching) & set(real_matching))


def compute_blocking_pairs(matching, student_prefs, college_priorities, capacities):
    matching_dict = dict(matching)
    assignments = defaultdict(list)

    counted = set()
    blocks = 0

    for stud, col in matching:
        if col != 'nothing':
            assignments[col].append(stud)

            # hallucination #1: student assigned to a college that doesn't rank them
            if stud not in college_priorities.get(col, []):
                if stud not in counted:
                    blocks += 1
                    counted.add(stud)

            # hallucination #2: student doesn't exist in the instance's preferences
            if stud not in student_prefs:
                if stud not in counted:
                    blocks += 1
                    counted.add(stud)

    for stud, prefs in student_prefs.items():
        if stud in counted:
            continue

        curr = matching_dict.get(stud, 'nothing')
        for pref in prefs:
            if pref == curr:
                break
            if stud not in college_priorities.get(pref, []):
                continue
            assigned = assignments.get(pref, [])
            if len(assigned) < capacities.get(pref, 0):  # school has a vacant seat
                blocks += 1
                counted.add(stud)
                break

            pref_list = college_priorities.get(pref, [])
            worst = max(assigned, key=lambda s: (pref_list.index(s) if s in pref_list else float('inf')))
            s_rank = pref_list.index(stud)
            worst_rank = pref_list.index(worst) if worst in pref_list else float('inf')
            if s_rank < worst_rank:  # they both prefer each other
                blocks += 1
                counted.add(stud)
                break

    return blocks


def evaluate_matching(llm_matching, real_matching, capacities, student_prefs, college_priorities):
    feas = compute_feasibility(llm_matching, capacities)
    correct = compute_correct_pairs(llm_matching, real_matching)
    assign_blocks = compute_blocking_pairs(llm_matching, student_prefs, college_priorities, capacities)
    if feas > 0:
        match_blocks = -1
    else:
        match_blocks = compute_blocking_pairs(llm_matching, student_prefs, college_priorities, capacities)

    return {
        'overcapacity': feas,
        'is_feasible': feas == 0,
        'assignment_blocking_pairs': assign_blocks,
        'is_assignment_stable': assign_blocks == 0,
        'matching_blocking_pairs': "N/A due to infeasibility" if feas > 0 else match_blocks,
        'is_matching_stable': False if feas > 0 else (match_blocks == 0),
        'correct_pairs': correct,
        'is_student_optimal': correct == len(student_prefs)
    }


def evaluate_from_files(llm_text, instance_path, real_match_path, real_is_list_format=True):
    llm_matching = extract_matching_from_text(llm_text)
    if llm_matching == []:
        return []
    capacities, student_prefs, college_priorities = parse_instance(instance_path)
    real_matching = read_matching(real_match_path, is_list_format=real_is_list_format)
    return evaluate_matching(llm_matching, real_matching, capacities, student_prefs, college_priorities)


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

    return text.strip()


current_path = os.getcwd()
home_path = base_path = os.path.join(*current_path.split(os.sep)[:5])


#############################################################################################
##############################   PARSER ARGUMENTS   #########################################
parser = argparse.ArgumentParser(description="Iterative-prompting LLM generation for the many-to-one matching problem")
parser.add_argument("--model", default='llama3-big',
                     help="LLMs model for generation. Expected values : llama3-big | llama3-instruct | "
                          "mistral-instruct | qwen2-7b-instruct | qwq-32b")
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
    model_name = 'Qwen/QwQ-32B '
elif args.model == 'mistral-instruct':
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
else:
    print('Please provide a valid entry for models -- ')
    raise KeyError

if args.model in ['qwen2-7b-instruct', 'mistral-instruct', 'qwq-32b']:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir, trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)


big_model_list = ['llama3-big', 'gemma-2-big', 'falcon-big', 'qwen2-7b-instruct', 'mistral-instruct', 'qwq-32b']

max_len = args.model_token_len
if args.model in ['qwq-32b']:
    max_len = max(max_len, 40960)

if args.model in big_model_list:
    dtype_m = 'float16'
    if args.model == 'gemma-2-big':
        dtype_m = "bfloat16"
    model = LLM(model=model_name, revision='main', max_model_len=max_len, max_num_batched_tokens=max_len,
                tokenizer=model_name, download_dir=args.cache_dir, trust_remote_code=True,
                dtype=dtype_m, tensor_parallel_size=args.nb_gpus)
    print('BIG MODEL')
else:
    model = LLM(model=model_name, revision='main', max_model_len=max_len, max_num_batched_tokens=max_len,
                tokenizer=model_name, download_dir=args.cache_dir, trust_remote_code=True, dtype='float16')


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

if args.model in ['qwq-32b']:
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=configs['temp_value'],
        top_p=configs['top_p_value'],
        top_k=configs['top_k_value'],
        repetition_penalty=configs['repet_value'],
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

last_generation_dir = os.path.join(generation_dir, 'last')
best_generation_dir = os.path.join(generation_dir, 'best')
no_generation_dir = os.path.join(generation_dir, 'no')
logs_dir = os.path.join(generation_dir, 'logs')

os.makedirs(last_generation_dir, exist_ok=True)
os.makedirs(best_generation_dir, exist_ok=True)
os.makedirs(no_generation_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)


for instruction_file in tqdm(instruction_files, desc="Instructions"):
    instruction_path = os.path.join(instruction_dir, instruction_file)
    with open(instruction_path, 'r', encoding='utf-8') as f:
        instruction_text = f.read()

    instruction_name = os.path.splitext(instruction_file)[0]
    needs_example = "[ADD EXAMPLE HERE]" in instruction_text

    for instance_file in tqdm(instance_files, desc=f"Instances for {instruction_name}", leave=False):
        instance_base = os.path.splitext(instance_file)[0]
        output_filename = f"generation_{instruction_name}_{instance_base}.txt"

        last_output_path = os.path.join(last_generation_dir, output_filename)
        best_output_path = os.path.join(best_generation_dir, output_filename)
        no_output_path = os.path.join(no_generation_dir, output_filename)
        logs_path = os.path.join(logs_dir, output_filename)

        if os.path.exists(last_output_path) and os.path.exists(best_output_path) and os.path.exists(no_output_path):
            print(f" --> Skipping {output_filename}, already generated.")
            continue
        else:
            print("--------------------------------- INSTRUCTION -----------------------------")
            print(output_filename)

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

        ########################## ITERATIVE PROMPTING #######################
        prompt = prompt.replace("[ADD INSTANCE HERE]", instance_text)

        max_iters = 5
        iter_count = 0
        metrics_satisfied = False
        last_answer = ""
        last_metrics = None
        augmented_prompt = prompt
        best_true_count = -1
        best_matching_str = ""
        last_matching_str = ""
        no_matching_str = ""
        logs = ""
        best_iter = 0

        while iter_count < max_iters and not metrics_satisfied:
            if args.model in ['qwq-32b']:
                answer = generate_with_qwq_vllm(
                    model, tokenizer, augmented_prompt, sampling_params,
                    strip_reasoning=True,
                    stop_words=None
                )
            else:
                outputs = model.generate(augmented_prompt, sampling_params=sampling_params, use_tqdm=False)
                answer = outputs[0].outputs[0].text.strip()
            last_answer = answer

            iteration = iter_count + 1
            logs += f"ITERATION NUMBER {iteration}\n"

            REAL_MATCH_folder = # path to dataset_match
            real_match_file = "match_" + os.path.basename(instance_path)
            real_match_path = os.path.join(REAL_MATCH_folder, real_match_file)
            metrics = evaluate_from_files(
                answer, instance_path,
                real_match_path,
                real_is_list_format=True
            )
            last_metrics = metrics

            # Always extract and capture iter 0 output, regardless of metrics
            extracted_matching = str(extract_matching_from_text(answer))
            if iter_count == 0:
                no_matching_str = extracted_matching

            if metrics == []:
                print("no matching")
                print(answer)
                print("-------------------------------------------------------------")
                not_good_lines = []
                not_good_lines.append("Your last attempt did not include a matching. Make sure to include it.")
            else:
                is_feasible = metrics.get("is_feasible", False)
                is_assignment_stable = metrics.get("is_assignment_stable", False)
                is_matching_stable = metrics.get("is_matching_stable", False)
                is_student_optimal = metrics.get("is_student_optimal", False)

                num_true = (is_feasible, is_assignment_stable, is_matching_stable, is_student_optimal).count(True)
                logs += f"This matching metrics results (feasibility,assignment stability, matching stability, optimality) is ({is_feasible},{is_assignment_stable},{is_matching_stable},{is_student_optimal})\n"

                metrics_satisfied = all([is_feasible, is_assignment_stable, is_matching_stable, is_student_optimal])

                if metrics_satisfied:
                    print("!!! ALL METRICS SATISFIED !!!")
                    best_true_count = 4
                    best_matching_str = str(extract_matching_from_text(answer))
                    extracted_matching = best_matching_str
                    last_matching_str = best_matching_str
                    logs += f"Optimal matching found at iteration {iteration}. Best and Last matching are both at iteration {iter_count}\n"
                    best_iter = iter_count
                    break

                last_matching_str = extracted_matching

                if num_true >= best_true_count:
                    if num_true == best_true_count:
                        logs += f"Found an equivalent matching (same number of metrics satisfied) at iteration {iteration}\n"
                    else:
                        logs += f"Found a better matching at iteration {iteration}, with {num_true} metrics satisfied instead of {best_true_count}\n"
                    best_true_count = num_true
                    best_matching_str = extracted_matching
                    best_iter = iter_count

                # Prepare instruction about which metrics failed
                not_good_lines = []
                not_good_lines.append("Attempted Response:")
                not_good_lines.append(extracted_matching)
                not_good_lines.append("This matching is incorrect. Here is some feedback based on four metrics:")
                not_good_lines.append(f"- Feasibility: {'The matching is feasible.' if is_feasible else 'The matching is not feasible. Capacity constraints are not respected.'} (number of seats over capacity={metrics['overcapacity']})")
                not_good_lines.append(f"- Assignment stability: {'The matching is assignment stable.' if is_assignment_stable else 'The matching is not assignment stable. There is at least one blocking pair.'} (number of blocking pairs={metrics['assignment_blocking_pairs']})")
                not_good_lines.append(f"- Matching stability: {'The matching is matching stable.' if is_matching_stable else 'The matching is not matching stable. There is at least one blocking pair.'} (number of blocking pairs={metrics['matching_blocking_pairs']})")
                not_good_lines.append(f"- Student-optimality: {'The matching is student-optimal.' if is_student_optimal else 'The matching is not student optimal. Some students do not have their best matching of all stable outcomes.'} (proportion of correct pairs={metrics['correct_pairs']}/{len(parse_instance(instance_path)[1])})")
                not_good_lines.append("Please provide a new matching, making sure that all metrics are satisfied.")

            # Insert feedback before the last line of the prompt
            prompt_lines = augmented_prompt.strip().split('\n')
            insert_pos = -1 if len(prompt_lines) >= 2 else len(prompt_lines)
            new_prompt_lines = prompt_lines[:insert_pos] + not_good_lines + prompt_lines[insert_pos:]
            augmented_prompt = '\n'.join(new_prompt_lines)

            iter_count += 1

            logs += f"End of iteration {iter_count}. The first best matching was found at iteration {best_iter}\n"

        # Write all three outputs
        with open(no_output_path, 'w', encoding='utf-8') as f:
            f.write(no_matching_str)

        with open(last_output_path, 'w', encoding='utf-8') as f:
            f.write(last_matching_str)

        with open(best_output_path, 'w', encoding='utf-8') as f:
            f.write(best_matching_str)

        with open(logs_path, 'w', encoding='utf-8') as f:
            f.write(logs)

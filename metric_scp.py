import ast
import os
import re
from collections import defaultdict

def read_matching(file_path, is_list_format=False):
    matchings = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if is_list_format:
            return ast.literal_eval(lines[0])
        for line in lines:
            match = re.match(r"\(\s*['\"]?(s\d+)['\"]?\s*,\s*['\"]?(c\d+|nothing)['\"]?\s*\)", line)
            if match:
                matchings.append((match.group(1), match.group(2)))
    return matchings

def parse_instance(file_path):
    capacities = {}
    student_prefs = {}
    college_priorities = {}

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    while lines and lines[0].startswith("c"):
        college, cap = lines.pop(0).split()
        capacities[college] = int(cap)

    while lines and lines[0].startswith("s"):
        parts = lines.pop(0).split()
        student = parts[0]
        prefs = [p.split(",")[1][:-1] for p in parts[1:]]
        student_prefs[student] = prefs

    for line in lines:
        parts = line.split()
        college = parts[0]
        priorities = [p.split(",")[1][:-1] for p in parts[1:]]
        college_priorities[college] = priorities

    return capacities, student_prefs, college_priorities

def compute_feasibility(matching, capacities):
    assigned = defaultdict(int)
    for _, college in matching:
        if college != "nothing":
            assigned[college] += 1
    return sum(max(0, assigned[c] - capacities.get(c, 0)) for c in assigned)

def compute_correct_pairs(llm_matching, real_matching):
    return len(set(llm_matching) & set(real_matching))

def compute_blocking_pairs(matching, student_prefs, college_priorities, capacities):
    matching_dict = dict(matching)
    college_assignments = defaultdict(list)
    for student, college in matching:
        if college != "nothing":
            college_assignments[college].append(student)

    blocking_pairs = 0
    for student, prefs in student_prefs.items():
        current_college = matching_dict.get(student, "nothing")
        for preferred_college in prefs:
            if preferred_college == current_college:
                break
            priority = college_priorities[preferred_college]
            assigned = [s for s in college_assignments[preferred_college] if s in priority]
            if student not in priority:
                continue
            if len(assigned) < capacities[preferred_college]:
                blocking_pairs += 1
                break
            if assigned:
                worst_student = max(assigned, key=lambda s: priority.index(s))
                if priority.index(student) < priority.index(worst_student):
                    blocking_pairs += 1
                    break
    return blocking_pairs

def run_batch(instance_dir, llm_match_dir, real_match_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for llm_file in os.listdir(llm_match_dir):
        if not llm_file.endswith("_extract.txt"):
            continue

        match = re.search(r"scp_\([^)]+\)_\w+_\d+_seed\d+", llm_file)
        if not match:
            print(f"Could not extract match pattern from: {llm_file}")
            continue

        base_core = match.group(0)
        instruction = llm_file.split(f"{base_core}")[0].rstrip("_")

        instance_file = f"{base_core}.txt"
        real_file = f"match_{base_core}.txt"
        output_file = f"metric_{instruction}_{base_core}_extract.txt"

        instance_path = os.path.join(instance_dir, instance_file)
        real_match_path = os.path.join(real_match_dir, real_file)
        llm_match_path = os.path.join(llm_match_dir, llm_file)
        output_path = os.path.join(output_dir, output_file)

        if not os.path.exists(instance_path):
            print(f"!!!!! Missing instance file: {instance_path}")
            continue
        if not os.path.exists(real_match_path):
            print(f"!!!! Missing real matching file: {real_match_path}")
            continue
        if not os.path.exists(llm_match_path):
            print(f"!!!!! Missing LLM matching file: {llm_match_path}")
            continue

        llm_matching = read_matching(llm_match_path, is_list_format=False)
        if not llm_matching:
            print(f"!!! Empty or unreadable matching in: {llm_match_path}")
            continue
        
        capacities, student_prefs, college_priorities = parse_instance(instance_path)
        real_matching = read_matching(real_match_path, is_list_format=True)

        feasibility = compute_feasibility(llm_matching, capacities)
        correct_pairs = compute_correct_pairs(llm_matching, real_matching)
        assignment_blocking_pairs = compute_blocking_pairs(llm_matching, student_prefs, college_priorities, capacities)

        if feasibility > 0: # there is overcapacity
            matching_blocking_pairs = -1  # invalid due to infeasibility
            is_matching_stable = False
        else:
            matching_blocking_pairs = compute_blocking_pairs(llm_matching, student_prefs, college_priorities, capacities)
            is_matching_stable = matching_blocking_pairs == 0

        with open(output_path, 'w') as out:
            out.write("Metric Evaluation Results:\n")
            out.write(f"# of Overcapacity (feasibility): {feasibility}\n")
            out.write(f"The matching is feasible : {feasibility == 0}\n")
            out.write(f"# of Blocking Pairs (assignment stability): {assignment_blocking_pairs}\n")
            out.write(f"The matching is assignment stable : {assignment_blocking_pairs == 0}\n")
            out.write(f"# of Blocking Pairs (matching stability): {matching_blocking_pairs if matching_blocking_pairs >= 0 else 'N/A due to infeasibility'}\n")
            out.write(f"The matching is matching stable : {is_matching_stable}\n")
            out.write(f"# of Correct Pairs (student-optimality): {correct_pairs}\n")
            out.write(f"The matching is student-optimal : {correct_pairs == len(student_prefs)}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated matchings against ground-truth DA matchings (feasibility, stability, student-optimality)."
    )
    parser.add_argument("--instance-dir", required=True, help="Directory containing the SCP instance files")
    parser.add_argument("--llm-match-dir", required=True, help="Directory containing extracted LLM matchings (*_extract.txt)")
    parser.add_argument("--real-match-dir", required=True, help="Directory containing ground-truth DA matchings")
    parser.add_argument("--output-dir", required=True, help="Directory to write per-instance metric files to")
    args = parser.parse_args()

    run_batch(
        instance_dir=args.instance_dir,
        llm_match_dir=args.llm_match_dir,
        real_match_dir=args.real_match_dir,
        output_dir=args.output_dir,
    )

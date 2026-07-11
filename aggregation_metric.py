import argparse
import os
import re
from collections import defaultdict
import pandas as pd
 
 
def parse_metric_file(filepath, num_students):
    with open(filepath, 'r') as f:
        content = f.read()
 
    try:
        feasible = "True" in re.search(r"The matching is feasible.*?: (\w+)", content).group(1)
        assignment_stable = "True" in re.search(r"The matching is assignment stable.*?: (\w+)", content).group(1)
        matching_stable = "True" in re.search(r"The matching is matching stable.*?: (\w+)", content).group(1)
        optimal = "True" in re.search(r"The matching is student-optimal.*?: (\w+)", content).group(1)
        return feasible, assignment_stable, matching_stable, optimal
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
        return None
 
 
def aggregate_metrics(metric_folder, type_str):
    """Aggregate metrics for each number of students and each preference type."""
    file_pattern = re.compile(
        rf"metric_generation_{type_str}_scp_\((?P<students>\d+),(?P<schools>\d+)\)_(?P<pref>[A-Za-z]+)_(?P<cap>\d+)_seed(?P<seed>\d+)_extract\.txt"
    )
 
    agg_by_students = defaultdict(lambda: defaultdict(int))
    agg_by_pref = defaultdict(lambda: defaultdict(int))
    agg_overall = defaultdict(int)
 
    for fname in os.listdir(metric_folder):
        if not fname.startswith(f"metric_generation_{type_str}_") or not fname.endswith(".txt"):
            continue
        match = file_pattern.match(fname)
        if not match:
            continue
 
        meta = match.groupdict()
        students = int(meta["students"])
        pref = meta["pref"].lower()
        full_path = os.path.join(metric_folder, fname)
 
        parsed = parse_metric_file(full_path, students)
        if parsed is None:
            continue
        feasible, assignment_stable, matching_stable, optimal = parsed
 
        # students
        agg_by_students[students]["count"] += 1
        agg_by_students[students]["assignment_stable_true"] += int(assignment_stable)
        agg_by_students[students]["matching_stable_true"] += int(matching_stable)
        agg_by_students[students]["feasible_true"] += int(feasible)
        agg_by_students[students]["optimal_true"] += int(optimal)
 
        # preference type
        agg_by_pref[pref]["count"] += 1
        agg_by_pref[pref]["assignment_stable_true"] += int(assignment_stable)
        agg_by_pref[pref]["matching_stable_true"] += int(matching_stable)
        agg_by_pref[pref]["feasible_true"] += int(feasible)
        agg_by_pref[pref]["optimal_true"] += int(optimal)
 
        # overall
        agg_overall["count"] += 1
        agg_overall["assignment_stable_true"] += int(assignment_stable)
        agg_overall["matching_stable_true"] += int(matching_stable)
        agg_overall["feasible_true"] += int(feasible)
        agg_overall["optimal_true"] += int(optimal)
 
    df_students = pd.DataFrame([
        {
            "students": k,
            "count": v["count"],
            "feasible_prop": v["feasible_true"] / v["count"],
            "assignment_stable_prop": v["assignment_stable_true"] / v["count"],
            "matching_stable_prop": v["matching_stable_true"] / v["count"],
            "optimal_prop": v["optimal_true"] / v["count"],
        }
        for k, v in sorted(agg_by_students.items())
    ])
 
    df_prefs = pd.DataFrame([
        {
            "preference_type": k,
            "count": v["count"],
            "feasible_prop": v["feasible_true"] / v["count"],
            "assignment_stable_prop": v["assignment_stable_true"] / v["count"],
            "matching_stable_prop": v["matching_stable_true"] / v["count"],
            "optimal_prop": v["optimal_true"] / v["count"],
        }
        for k, v in sorted(agg_by_pref.items())
    ])
 
    df_overall = pd.DataFrame([{
        "count": agg_overall["count"],
        "feasible_prop": agg_overall["feasible_true"] / agg_overall["count"],
        "assignment_stable_prop": agg_overall["assignment_stable_true"] / agg_overall["count"],
        "matching_stable_prop": agg_overall["matching_stable_true"] / agg_overall["count"],
        "optimal_prop": agg_overall["optimal_true"] / agg_overall["count"],
    }])
 
    return df_students, df_prefs, df_overall
 
 
def run_aggregation(metric_folder, output_folder, output_filename, type_str):
    """Aggregate one metric_folder and write the three summary tables to a single .txt file."""
    os.makedirs(output_folder, exist_ok=True)
 
    df_students, df_prefs, df_overall = aggregate_metrics(metric_folder, type_str)
 
    output_path = os.path.join(output_folder, f"{output_filename}.txt")
 
    with open(output_path, 'w') as f:
        f.write("Aggregated Metrics by Number of Students:\n\n")
        f.write(df_students.to_string(index=False))
        f.write("\n\nAggregated Metrics by Preference Type:\n\n")
        f.write(df_prefs.to_string(index=False))
        f.write("\n\nAggregated Metrics Overall:\n\n")
        f.write(df_overall.to_string(index=False))
 
    print(f"Aggregation completed. Results saved at {output_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-instance metric files into summary tables (by student count, by preference type, and overall)."
    )
    parser.add_argument("--metric-folder", required=True, help="Folder containing metric_generation_{type_str}_*.txt files")
    parser.add_argument("--output-folder", required=True, help="Folder to write the aggregated results file to")
    parser.add_argument("--output-filename", required=True, help="Output filename, without .txt extension")
    parser.add_argument(
        "--type-str", required=True,
        help='Generation type tag matching the metric filenames, e.g. "vague", "basic", "basic_role", '
             '"CoT_pseudo", "CoT_python", "CoT_txt", "CoT_unsupervised", "ICL_1", "ICL_steps"'
    )
    args = parser.parse_args()
 
    run_aggregation(args.metric_folder, args.output_folder, args.output_filename, args.type_str)

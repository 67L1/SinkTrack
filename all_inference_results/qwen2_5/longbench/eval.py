import json
from collections import defaultdict


def analyze_length_results(result_data):
    length_scores = defaultdict(list)
    all_scores = []

    for task_name, scores_by_length in result_data.items():
        if isinstance(scores_by_length, dict):
            for length_bucket, score in scores_by_length.items():
                length_scores[length_bucket].append(score)
                all_scores.append(score)

    length_averages = {}
    for length_bucket, scores in length_scores.items():
        if scores:
            average = sum(scores) / len(scores)
            length_averages[length_bucket] = round(average, 2)
        else:
            length_averages[length_bucket] = 0.0

    total_average = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print("--- LongBench-E length evaluation results ---")
    print("-" * 35)
    print(f"{'Length Range':<15} | {'AVG':<10}")
    print("-" * 35)

    for bucket in ["0-4k", "4-8k", "8k+"]:
        if bucket in length_averages:
            print(f"{bucket:<15} | {length_averages[bucket]:<10.2f}")

    print("-" * 35)
    print(f"{'Overall Average':<15} | {total_average:<10.2f}")
    print("-" * 35)


path_list = [
    "direct_42/result_e.json",
    "cot_42/result_e.json",
    "sinktrack_323/result_e.json",
]

for path in path_list:
    print(path)

    with open(path, 'r') as f:
        result_data = json.load(f)


    analyze_length_results(result_data)
import json
import string
import re
from collections import Counter, defaultdict
import numpy as np
import os

BREAK_TOTAL = 99999


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_f1_for_item(predicted_answer, ground_truth_answers):
    if not ground_truth_answers:
        return 0.0
    if ground_truth_answers[0] == 'CANNOTANSWER':
        ground_truth_answers.append('not provide')
    if 'not provide' in predicted_answer:
        predicted_answer = 'not provide'
    scores = [f1_score(predicted_answer, gt) for gt in ground_truth_answers]
    return max(scores)


def calculate_overall_f1(input_file_path):
    total_items = 0
    all_f1_scores = []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            predicted_answer = item.get("predicted_answer", "")
            ground_truth_answers = item.get("ground_truth_answers", [])
            item_f1_score = calculate_f1_for_item(predicted_answer, ground_truth_answers)
            all_f1_scores.append(item_f1_score)
            total_items += 1
            if total_items >= BREAK_TOTAL: break

    if total_items > 0:
        average_f1 = sum(all_f1_scores) / total_items
        return average_f1 * 100
    return 0.0


def calculate_my_acc(input_file_path):
    total = 0
    acc = 0
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            predicted_answer = item.get("predicted_answer", "")
            ground_truth_answers = item.get("ground_truth_answers", [])
            total += 1

            matched = False
            if not ground_truth_answers:
                if "not provide" in predicted_answer.lower() or "cannotanswer" in predicted_answer.lower():
                    matched = True
            else:
                for ans in ground_truth_answers:
                    clean_pred = predicted_answer.replace(',', '').replace('.', '')
                    clean_ans = ans.replace(',', '').replace('.', '')
                    if clean_ans.lower() in clean_pred.lower() or clean_pred.lower() in clean_ans.lower():
                        matched = True
                        break

            if matched:
                acc += 1
            else:
                if "I cannot answer the question based on the provided text." in predicted_answer:
                    total -= 1
                    continue

            if total >= BREAK_TOTAL: break

    accuracy = (acc / total) if total > 0 else 0
    return accuracy * 100



def main():
    METHODS = ['direct', 'cot', 'sinktrack']

    SEEDS = [323, 500, 900]

    all_results = defaultdict(lambda: defaultdict(list))

    for method in METHODS:
        print(f"\n===== Evaluating Method: {method} =====")
        for seed in SEEDS:
            file_path = f"{method}_{seed}.jsonl"
            print(f"--> Processing file: {file_path}")

            if not os.path.exists(file_path):
                print(f"    [Warning] File not found, skipping.")
                continue

            overall_f1 = calculate_overall_f1(file_path)
            my_acc = calculate_my_acc(file_path)

            print(f"    Overall F1: {overall_f1:.2f}, my_ACC: {my_acc:.2f}")

            all_results[method]["f1s"].append(overall_f1)
            all_results[method]["accs"].append(my_acc)

    print("\n\n================================================")
    print("           Final Aggregated Results           ")
    print("================================================")

    for method in METHODS:
        f1_scores = all_results[method]["f1s"]
        acc_scores = all_results[method]["accs"]

        if not f1_scores:
            print(f"\n--- Method: {method} ---")
            print("  No results found for this method.")
            continue

        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        acc_mean = np.mean(acc_scores)
        acc_std = np.std(acc_scores)

        print(f"\n--- Method: {method} ---")

        f1_scores_str = [f"{s:.2f}" for s in f1_scores]
        acc_scores_str = [f"{s:.2f}" for s in acc_scores]

        print(f"  Individual Overall F1s : {f1_scores_str}")
        print(f"  => Overall F1 (Mean ± Std): {f1_mean:.2f} ± {f1_std:.2f}")

        print(f"  Individual my_ACCs     : {acc_scores_str}")
        print(f"  => my_ACC (Mean ± Std)    : {acc_mean:.2f} ± {acc_std:.2f}")

    print("\n================================================")


if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is not installed. Please install it using 'pip install numpy'")
        exit()

    main()
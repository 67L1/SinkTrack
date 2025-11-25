import json
import string
import re
from collections import Counter, defaultdict
import numpy as np
import os



def is_overlapping(x1, x2, y1, y2):
    return max(x1, y1) <= min(x2, y2)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

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
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def display_counter(title, c, c2=None):
    print(title)
    for key, _ in c.most_common():
        if c2:
            print('%s: %d / %d, %.1f%%, F1: %.1f' % (
                key, c[key], sum(c.values()), c[key] * 100. / sum(c.values()), sum(c2[key]) * 100. / len(c2[key])))
        else:
            print('%s: %d / %d, %.1f%%' % (key, c[key], sum(c.values()), c[key] * 100. / sum(c.values())))


def leave_one_out_max(prediction, ground_truths, article):
    if len(ground_truths) == 1:
        return metric_max_over_ground_truths(prediction, ground_truths, article)[1]
    else:
        t_f1 = []
        for i in range(len(ground_truths)):
            idxes = list(range(len(ground_truths)))
            idxes.pop(i)
            refs = [ground_truths[z] for z in idxes]
            t_f1.append(metric_max_over_ground_truths(prediction, refs, article)[1])
        return 1.0 * sum(t_f1) / len(t_f1)


def metric_max_over_ground_truths(prediction, ground_truths, article):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = compute_span_overlap(prediction, ground_truth, article)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths, key=lambda x: x[1])


def handle_cannot(refs):
    num_cannot = 0
    num_spans = 0
    for ref in refs:
        if ref == 'CANNOTANSWER':
            num_cannot += 1
        else:
            num_spans += 1
    if num_cannot >= num_spans:
        refs = ['CANNOTANSWER']
    else:
        refs = [x for x in refs if x != 'CANNOTANSWER']
    return refs


def leave_one_out(refs):
    if len(refs) == 1:
        return 1.
    splits = []
    for r in refs:
        splits.append(r.split())
    t_f1 = 0.0
    for i in range(len(refs)):
        m_f1 = 0
        for j in range(len(refs)):
            if i == j:
                continue
            f1_ij = f1_score(refs[i], refs[j])
            if f1_ij > m_f1:
                m_f1 = f1_ij
        t_f1 += m_f1
    return t_f1 / len(refs)


def compute_span_overlap(pred_span, gt_span, text):
    if gt_span == 'CANNOTANSWER':
        if pred_span == 'CANNOTANSWER':
            return 'Exact match', 1.0
        return 'No overlap', 0.
    fscore = f1_score(pred_span, gt_span)
    pred_start = text.find(pred_span)
    gt_start = text.find(gt_span)
    if pred_start == -1 or gt_start == -1:
        return 'Span indexing error', fscore
    pred_end = pred_start + len(pred_span)
    gt_end = gt_start + len(gt_span)
    fscore = f1_score(pred_span, gt_span)
    overlap = is_overlapping(pred_start, pred_end, gt_start, gt_end)
    if exact_match_score(pred_span, gt_span):
        return 'Exact match', fscore
    if overlap:
        return 'Partial overlap', fscore
    else:
        return 'No overlap', fscore


def eval_fn(val_results, model_results, min_f1=0.4, verbose=False):
    my_acc = 0
    my_total = 0
    span_overlap_stats = Counter()
    total_qs = 0.
    f1_stats = defaultdict(list)

    for p in val_results:
        for par in p['paragraphs']:
            did = par['id']
            qa_list = par['qas']
            for qa in qa_list[:3]:
                q_idx = qa['id']
                val_spans = [anss['text'] for anss in qa['answers']]
                val_spans = handle_cannot(val_spans)
                hf1 = leave_one_out(val_spans)

                if did not in model_results or q_idx not in model_results[did]:
                    if hf1 >= min_f1:
                        total_qs += 1
                    continue

                pred_span, _, _ = model_results[did][q_idx]

                my_total += 1
                ok = False
                # Clean prediction for matching
                clean_pred = pred_span.replace('.', '').replace(',', '').lower()
                for a in val_spans:
                    clean_a = a.replace('.', '').replace(',', '').lower()
                    if clean_a in clean_pred or clean_pred in clean_a:
                        my_acc += 1
                        ok = True
                        break

                if hf1 < min_f1:
                    continue

                max_overlap, _ = metric_max_over_ground_truths(pred_span, val_spans, par['context'])
                max_f1 = leave_one_out_max(pred_span, val_spans, par['context'])

                span_overlap_stats[max_overlap] += 1
                f1_stats[max_overlap].append(max_f1)
                total_qs += 1.

    all_f1s = sum(f1_stats.values(), [])
    overall_f1 = 100.0 * sum(all_f1s) / len(all_f1s) if all_f1s else 0
    my_acc_score = 100.0 * my_acc / my_total if my_total > 0 else 0

    print(f'Overall F1: {overall_f1:.2f}')
    print(f"my ACC: {my_acc} / {my_total} = {my_acc_score:.2f}")
    print("-" * 20)

    return {"overall_f1": overall_f1, "my_acc": my_acc_score}



def run_single_evaluation(val_file_path, model_output_path):
    if not os.path.exists(model_output_path):
        print(f"Warning: File not found, skipping: {model_output_path}")
        return None

    with open(val_file_path, 'r') as f:
        val_data = json.load(f)['data']

    preds = defaultdict(dict)
    with open(model_output_path, 'r') as f:
        for line in f:
            if line.strip():
                pred_idx = json.loads(line.strip())
                dia_id = pred_idx['qid'][0].split("_q#")[0]
                for qid, qspan, qyesno, qfollowup in zip(pred_idx['qid'], pred_idx['best_span_str'], pred_idx['yesno'],
                                                         pred_idx['followup']):
                    preds[dia_id][qid] = qspan, qyesno, qfollowup

    metrics = eval_fn(val_data, preds)
    return metrics


def main():
    VAL_FILE_PATH = 'val_v0.2.json'
    METHODS = ['direct', 'cot', 'sinktrack']

    SEEDS = [323, 500, 900]

    if not os.path.exists(VAL_FILE_PATH):
        print(f"Error: Validation file not found at '{VAL_FILE_PATH}'")
        print("Please make sure the validation file is in the same directory and the path is correct.")
        return

    all_results = defaultdict(lambda: defaultdict(list))

    for method in METHODS:
        print(f"\n===== Evaluating Method: {method} =====")
        for seed in SEEDS:
            model_output_file = f"{method}_{seed}.jsonl"
            print(f"Running for: {model_output_file}")

            result = run_single_evaluation(VAL_FILE_PATH, model_output_file)

            if result:
                all_results[method]["f1s"].append(result["overall_f1"])
                all_results[method]["accs"].append(result["my_acc"])

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
    main()
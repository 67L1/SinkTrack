import json
import os
import yaml
import string
import re
import numpy as np
from collections import Counter, defaultdict
from argparse import ArgumentParser


def is_overlapping(x1, x2, y1, y2):
    return max(x1, y1) <= min(x2, y2)


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


my_acc = 0
my_total = 0


def load_predictions(model_output_path):
    preds = defaultdict(dict)
    total = 0
    with open(model_output_path, 'r') as f:
        for line in f:
            if line.strip():
                pred_idx = json.loads(line.strip())
                dia_id = pred_idx['qid'][0].split("_q#")[0]
                for qid, qspan, qyesno, qfollowup in zip(
                    pred_idx['qid'], pred_idx['best_span_str'],
                    pred_idx['yesno'], pred_idx['followup']
                ):
                    preds[dia_id][qid] = qspan, qyesno, qfollowup
                    total += 1
    return preds, total


def eval_fn(val_results, model_results, min_f1, verbose, max_turn=None):
    global my_acc, my_total
    span_overlap_stats = Counter()
    total_qs = 0.
    f1_stats = defaultdict(list)
    unfiltered_f1s = []
    human_f1 = []
    HEQ = 0.
    DHEQ = 0.
    total_dials = 0.
    yes_nos = []
    followups = []
    unanswerables = []

    for p in val_results:
        for par in p['paragraphs']:
            did = par['id']
            qa_list = par['qas']

            qa_subset = qa_list[:max_turn] if max_turn is not None else qa_list

            good_dial = 1.
            dialogue_evaluated = False

            for qa in qa_subset:
                dialogue_evaluated = True
                q_idx = qa['id']
                val_spans = [anss['text'] for anss in qa['answers']]
                val_spans = handle_cannot(val_spans)
                hf1 = leave_one_out(val_spans)

                if did not in model_results or q_idx not in model_results[did]:
                    if verbose:
                        print(did, q_idx, 'no prediction for this dialogue id')
                    good_dial = 0
                    f1_stats['NO ANSWER'].append(0.0)
                    yes_nos.append(False)
                    followups.append(False)
                    if val_spans == ['CANNOTANSWER']:
                        unanswerables.append(0.0)
                    total_qs += 1
                    unfiltered_f1s.append(0.0)
                    if hf1 >= min_f1:
                        human_f1.append(hf1)
                    continue

                pred_span, pred_yesno, pred_followup = model_results[did][q_idx]
                if '.' in pred_span:
                    pred_span = pred_span.replace('.', '')
                if ',' in pred_span:
                    pred_span = pred_span.replace(',', '')

                ok = False
                my_total += 1
                for a in val_spans:
                    if '.' in a:
                        a = a.replace('.', '')
                    if ',' in a:
                        a = a.replace(',', '')
                    if a.lower() in pred_span.lower() or pred_span.lower() in a.lower():
                        my_acc += 1
                        ok = True
                        break

                max_overlap, _ = metric_max_over_ground_truths(pred_span, val_spans, par['context'])
                max_f1 = leave_one_out_max(pred_span, val_spans, par['context'])
                unfiltered_f1s.append(max_f1)

                if hf1 < min_f1:
                    continue

                human_f1.append(hf1)
                yes_nos.append(pred_yesno == qa['yesno'])
                followups.append(pred_followup == qa['followup'])
                if val_spans == ['CANNOTANSWER']:
                    unanswerables.append(max_f1)

                if verbose:
                    print("-" * 20)
                    print(f"Evaluating QID: {q_idx}")
                    print(f"Prediction: {pred_span}")
                    print(f"Ground Truths: {val_spans}")
                    print(f"F1: {max_f1:.4f}, Human F1: {hf1:.4f}")
                    print("-" * 20)

                if max_f1 >= hf1:
                    HEQ += 1.
                else:
                    good_dial = 0.

                span_overlap_stats[max_overlap] += 1
                f1_stats[max_overlap].append(max_f1)
                total_qs += 1.

            if dialogue_evaluated:
                DHEQ += good_dial
                total_dials += 1

    DHEQ_score = 100.0 * DHEQ / total_dials if total_dials > 0 else 0
    HEQ_score = 100.0 * HEQ / total_qs if total_qs > 0 else 0
    all_f1s = sum(f1_stats.values(), [])
    overall_f1 = 100.0 * sum(all_f1s) / len(all_f1s) if all_f1s else 0
    unfiltered_f1 = 100.0 * sum(unfiltered_f1s) / len(unfiltered_f1s) if unfiltered_f1s else 0
    yesno_score = (100.0 * sum(yes_nos) / len(yes_nos)) if yes_nos else 0
    followup_score = (100.0 * sum(followups) / len(followups)) if followups else 0
    unanswerable_score = (100.0 * sum(unanswerables) / len(unanswerables)) if unanswerables else 0

    metric_json = {"unfiltered_f1": unfiltered_f1, "f1": overall_f1, "HEQ": HEQ_score, "DHEQ": DHEQ_score,
                   "yes/no": yesno_score, "followup": followup_score, "unanswerable_acc": unanswerable_score}

    if verbose:
        print("=======================")
        display_counter('Overlap Stats', span_overlap_stats, f1_stats)

    print("=======================")
    print('Overall F1: %.2f' % overall_f1)
    print(f"my ACC: {my_acc} / {my_total} = {100 * float(my_acc) / my_total:.2f}%")
    print('Accuracy On Unanswerable Questions: {0:.1f} %% ({1:d} questions)'.format(unanswerable_score, len(unanswerables)))
    print('Model F1 >= Human F1 (Questions): %d / %d, %.1f%%' % (HEQ, total_qs, HEQ_score))
    print('Model F1 >= Human F1 (Dialogs - based on first 2 Qs): %d / %d, %.1f%%' % (DHEQ, total_dials, DHEQ_score))
    print("=======================")
    return metric_json


def print_summary(all_seed_metrics):
    """Print mean ± variance across seeds for each turn config, matching paper Table format."""
    turn_keys = list(all_seed_metrics[list(all_seed_metrics.keys())[0]].keys())
    metric_keys = ["f1", "HEQ", "DHEQ"]

    print("\n" + "=" * 60)
    print("  SUMMARY: Mean ± Variance across seeds")
    print("=" * 60)
    for turn_key in turn_keys:
        print(f"\n  {turn_key}")
        print("-" * 40)
        for mk in metric_keys:
            vals = [all_seed_metrics[seed][turn_key][mk] for seed in all_seed_metrics]
            mean = np.mean(vals)
            var  = np.var(vals)
            print(f"  {mk:10s}: {mean:.2f} ± {var:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    default_val_file = os.path.join(
        repo_root,
        "all_inference_codes",
        "datasets",
        "val_v0.2.json",
    )

    default_model_output = os.path.join(
        repo_root,
        "all_inference_results",
        "llama3_1",
        "quac",
        "sinktrack_{seed}.jsonl",
    )

    baseline_to_cfg_key = {
        "sinktrack": "sinktrack_output_path",
        "cot": "cot_output_path",
        "direct": "direct_output_path",
    }

    parser = ArgumentParser(
        description=(
            "QuAC evaluation script for SinkTrack.\n"
            "Quick start (from this folder): `python new_scorer.py`\n"
            "By default it will use:\n"
            f"  val_file    = {default_val_file}\n"
            f"  model_output= {default_model_output}\n"
            f"  seeds       = {cfg['experiment']['seeds']}\n"
        )
    )
    parser.add_argument(
        '--baseline',
        type=str,
        choices=sorted(baseline_to_cfg_key.keys()),
        default='sinktrack',
        help=(
            "Which baseline to evaluate. "
            "Choices: {sinktrack, cot, direct}. "
            "If you do not pass --model_output explicitly, this flag "
            "controls which JSONL files (e.g. cot_{seed}.jsonl) are used "
            "under the output.base_dir specified in config.yaml."
        ),
    )
    parser.add_argument(
        '--val_file',
        type=str,
        default=default_val_file,
        help='Path to validation JSON file (default: repo val_v0.2.json)',
    )
    parser.add_argument(
        '--model_output',
        type=str,
        default=default_model_output,
        help=(
            'Path or template to model prediction JSONL. '
            'Supports {seed} placeholder, e.g. sinktrack_{seed}.jsonl. '
            'Default points to SinkTrack outputs under all_inference_results/llama3_1/quac/.'
        ),
    )
    parser.add_argument(
        '--o',
        type=str,
        required=False,
        help='Optional path to save metrics JSON (aggregated over seeds).',
    )
    parser.add_argument(
        '--min_f1',
        type=float,
        default=cfg["evaluation"]["min_f1"],
        help=f"Minimum human F1 to keep an example (default from config.yaml: {cfg['evaluation']['min_f1']}).",
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=cfg["evaluation"]["verbose"],
        help="Print per-question details (overrides config.yaml default).",
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=cfg["experiment"]["seeds"],
        help='Seeds to evaluate. Defaults to seeds in config.yaml.',
    )
    args = parser.parse_args()

    if args.model_output == default_model_output and args.baseline != "sinktrack":
        output_cfg = cfg.get("output", {})
        base_dir = output_cfg.get("base_dir", os.path.join(repo_root, "all_inference_results", "llama3_1", "quac"))
        if not os.path.isabs(base_dir):
            base_dir = os.path.normpath(os.path.join(repo_root, base_dir))

        cfg_key = baseline_to_cfg_key[args.baseline]
        stem = output_cfg.get(cfg_key)
        if stem is None:
            raise ValueError(
                f"Baseline '{args.baseline}' selected, but '{cfg_key}' "
                "is missing in config.yaml under 'output'."
            )

        name, ext = os.path.splitext(stem)
        if not ext:
            ext = ".jsonl"
        args.model_output = os.path.join(base_dir, f"{name}" + "_{seed}" + ext)

    if not os.path.exists(args.val_file):
        raise FileNotFoundError(
            f"Validation file not found: {args.val_file}\n"
            "If you placed val_v0.2.json elsewhere, pass it via --val_file."
        )

    print(f"Using validation file   : {args.val_file}")
    print(f"Using model_output spec : {args.model_output}")
    print(f"Baseline                : {args.baseline}")
    print(f"Seeds                   : {args.seeds}")
    print(f"min_f1                  : {args.min_f1}")
    print(f"verbose                 : {args.verbose}")

    val = json.load(open(args.val_file, 'r'))['data']

    turn_configs = [
        (1,    "Turn-1 (QuAC-1)"),
        (2,    "Turn-2 (QuAC-2)"),
        (3,    "Turn-3 (QuAC-3)"),
        (None, "All Turns (QuAC)"),
    ]

    all_seed_metrics = {}

    for seed in args.seeds:
        if "{seed}" in args.model_output:
            model_output_path = args.model_output.format(seed=seed)
        else:
            stem, ext = os.path.splitext(args.model_output)
            model_output_path = f"{stem}_{seed}{ext}"

        print(f"\n{'#' * 60}")
        print(f"  Seed: {seed}  |  File: {model_output_path}")
        print(f"{'#' * 60}")

        preds, total = load_predictions(model_output_path)
        seed_metrics = {}

        for max_turn, label in turn_configs:
            my_acc = 0
            my_total = 0
            print(f"\n{'=' * 40}")
            print(f"  Evaluation: {label}")
            print(f"{'=' * 40}")
            metrics = eval_fn(val, preds, args.min_f1, args.verbose, max_turn=max_turn)
            turn_key = f"turn_{max_turn}" if max_turn is not None else "all"
            seed_metrics[turn_key] = metrics

        all_seed_metrics[seed] = seed_metrics
        print(f"total: {total}")

    print_summary(all_seed_metrics)

    if args.o:
        with open(args.o, 'w') as fout:
            json.dump(all_seed_metrics, fout, indent=2)
        print(f"\nMetrics saved to {args.o}")
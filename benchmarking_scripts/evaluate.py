import argparse
import json
import re
from typing import Dict, Tuple, Optional

letter_switch = {
    '0': 'zero',
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five',
    '6': 'six',
    '7': 'seven',
    '8': 'eight',
    '9': 'nine',
}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"^[\s\*\_\-\=\:\|\"'`]+", "", s)
    s = re.sub(r"[\s\*\_\-\=\:\|\"'`\.!,;？。！、]+$", "", s)
    return s


def try_parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def extract_from_pred(pred_raw: str) -> str:
    if pred_raw is None:
        return ""
    pred = pred_raw.lower()

    tail = None
    idx = pred.find("answer:")
    if idx != -1:
        tail = pred[idx + len("answer:"):]
    else:
        idx2 = pred.find("answer")
        if idx2 != -1:
            tail = pred[idx2 + len("answer"):]
    if tail is None:
        tail = pred

    tail = re.sub(r"^\s*[:：=\-–—~>*()\[\]]*\s*(?:is|are|=)?\s*", "", tail)
    tail = tail.splitlines()[0] if tail else ""
    tail = normalize_text(tail)
    tail = re.sub(r"^[\*\_`]+|[\*\_`]+$", "", tail).strip()

    return tail


def parse_options_from_question(question: str) -> Dict[str, str]:
    options = {}
    pattern = re.compile(r"^\s*\(?([A-Z])\)?[.\s]\s*(.*)", re.MULTILINE)
    matches = pattern.findall(question)
    for match in matches:
        option_letter = match[0]
        option_text = match[1].strip()
        if len(option_letter) == 1 and option_text:
            options[option_letter] = option_text
    return options


def compare_pred_answer(pred_text: str, ans_text: str) -> bool:
    pred_norm = normalize_text(pred_text)
    ans_norm = normalize_text(ans_text)

    p_num = try_parse_number(pred_norm)
    a_num = try_parse_number(ans_norm)

    if p_num is not None and a_num is not None:
        return abs(p_num - a_num) < 1e-9

    if pred_norm == ans_norm:
        return True
    if ans_norm in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        correct_option_text = letter_switch[ans_norm]
        if correct_option_text in pred_norm or pred_norm in correct_option_text:
            return True
        elif correct_option_text == 'zero':
            correct_option_text = 'no'
            if correct_option_text in pred_norm or pred_norm in correct_option_text:
                return True

    if len(ans_norm) > 1:
        pattern = r'\b' + re.escape(ans_norm) + r'\b'
        if re.search(pattern, pred_norm):
            return True

    return False

from collections import defaultdict
def calculate_f1_macro(y_true, y_pred):
    labels = sorted(list(set(y_true) | set(y_pred)))

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            stats[true]["tp"] += 1
        else:
            stats[true]["fn"] += 1
            stats[pred]["fp"] += 1

    f1_scores = []
    for label in labels:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]

        if (tp + fp) == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if (tp + fn) == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1)

    if not f1_scores:
        return 0.0

    print(f"F1 (Macro): {sum(f1_scores) / len(f1_scores):.6f}")
    return sum(f1_scores) / len(f1_scores)


def main(input_file):
    total = 0
    correct = 0
    errors = []
    show_errors = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {ln}")
                continue

            total += 1

            ans = str(obj.get("answer", "")).strip()
            pred_raw = str(obj.get("pred", "")).strip()
            question = str(obj.get("question", "")).strip()

            pred_extracted = extract_from_pred(pred_raw)

            ok = False
            options_map = {}

            if compare_pred_answer(pred_extracted, ans):
                ok = True
            elif pred_extracted not in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                options_map = parse_options_from_question(question)
                ans_key = ans.upper()
                if ans_key in options_map:
                    correct_option_text = options_map[ans_key]
                    correct_option_text = correct_option_text.lower()

                    if correct_option_text in pred_extracted or pred_extracted in correct_option_text:
                        ok = True

                    if ok is False and correct_option_text in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                        correct_option_text = letter_switch[correct_option_text]
                        if correct_option_text in pred_extracted or pred_extracted in correct_option_text:
                            ok = True

                        elif correct_option_text == 'zero':
                            correct_option_text = 'no'
                            if correct_option_text in pred_extracted or pred_extracted in correct_option_text:
                                ok = True

            if ok:
                correct += 1
            else:
                if len(errors) < show_errors and ans not in ['a', 'b', 'c', 'd', 'e', 'f', 'g'] and pred_extracted not in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                    errors.append({
                        "line_no": ln,
                        "id": obj.get("id"),
                        "question": question,
                        "answer": ans,
                        "pred_raw": pred_raw,
                        "parsed_pred": pred_extracted,
                        "options_map": options_map,
                    })


    if show_errors > 0 and errors:
        print(f"\n--- Top {min(len(errors), show_errors)} Sample Errors ---")
        for e in errors:
            print(json.dumps(e, ensure_ascii=False, indent=2))
            print("-" * 20)

    acc = (correct / total) if total > 0 else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"ACC: {acc:.6f}")


if __name__ == "__main__":
    file = f"./sinktrack/qwen7b/mmstar/res_323.json"
    main(file)
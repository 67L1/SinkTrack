#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用法:
    python calc_acc.py
    (输入文件已在代码中硬编码)
"""

import argparse
import json
import re
from typing import Dict, Tuple, Optional
from collections import defaultdict

# <--- 修改开始 ---
# 引入 numpy 用于计算均值和标准差
import numpy as np


# <--- 修改结束 ---

# 全局变量 y_true 和 y_pred 已被移入 main 函数内部，以避免在处理多个文件时数据累积。

def calculate_f1_macro(y_true, y_pred):
    """
    计算宏平均F1分数 (Macro F1-Score)。
    """
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

    macro_f1 = sum(f1_scores) / len(f1_scores)
    # 打印单个文件的 F1 分数
    print(f"F1 (Macro): {macro_f1:.6f}")
    return macro_f1


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

    if pred_norm == ans_norm:
        return True


    return False


def recorder(line, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]

    line = line.replace('.', '')
    line = line.replace(',', '')
    words = line.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        pred_list.append(0)
    else:
        pred_list.append(1)

    return pred_list


# <--- 修改开始 ---
# 将原 main 函数重命名为 process_single_file，并让它返回 acc 和 f1
def process_single_file(input_file: str) -> Tuple[float, float]:
    """处理单个文件并返回其ACC和F1分数"""
    # 将 y_true 和 y_pred 移到函数内部，确保每次调用都重新初始化
    y_true = []
    y_pred = []

    total = 0
    correct = 0
    errors = []
    pred_list = []
    label_list = []
    show_errors = 0  # 如果需要显示错误，可以修改这个值

    with open(input_file, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()

            obj = json.loads(line)

            pred_extracted = None
            total += 1
            ans = str(obj.get("answer", "")).strip()
            pred_raw = str(obj.get("pred", "")).strip()
            question = str(obj.get("question", "")).strip()

            pred_list = recorder(pred_raw, pred_list)
            if ans.lower() == 'yes':
                label_list.append(1)
            elif ans.lower() == 'no':
                label_list.append(0)

    def print_acc(pred_list, label_list):
        pos = 1
        neg = 0
        yes_ratio = pred_list.count(1) / len(pred_list)
        # unknown_ratio = pred_list.count(2) / len(pred_list)

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(pred_list, label_list):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        print('TP\tFP\tTN\tFN\t')
        print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F1 score: {}'.format(f1))
        print('Yes ratio: {}'.format(yes_ratio))
        return acc, precision, recall, f1

    acc, precision, recall, f1 = print_acc(pred_list, label_list)
    return acc, precision, recall, f1



if __name__ == "__main__":
    # 定义要处理的文件列表
    input_files = ['res_323.json']

    # 用于存储每个文件的结果
    all_accuracies = []
    all_f1_scores = []

    # 循环处理每个文件
    for file_path in input_files:
        try:
            print(f"\n--- Processing file: {file_path} ---")
            # 调用处理函数，获取ACC和F1
            acc, precision, recall, f1 = process_single_file(file_path)
            # 将结果添加到列表中
            all_accuracies.append(acc)
            all_f1_scores.append(f1)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")

    # 检查是否成功处理了任何文件
    if all_accuracies and all_f1_scores:
        # 计算均值和标准差
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)

        mean_f1 = np.mean(all_f1_scores)
        std_f1 = np.std(all_f1_scores)

        # 打印最终的汇总结果
        print("\n" + "=" * 30)
        print("--- Overall Results ---")
        print("=" * 30)
        print(f"Processed {len(input_files)} files.")
        print(f"ACCURACY -> Mean: {mean_acc * 100:.2f}  Std: {std_acc * 100:.2f}")
        print(f"F1-Macro -> Mean: {mean_f1 * 100:.2f}  Std: {std_f1 * 100:.2f}")
        print("=" * 30)
    else:
        print("\nNo files were processed successfully. Cannot calculate overall results.")
# <--- 修改结束 ---
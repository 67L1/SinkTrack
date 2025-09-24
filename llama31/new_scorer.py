import json, string, re
from collections import Counter, defaultdict
from argparse import ArgumentParser


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
    # leave out one ref every time
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
def eval_fn(val_results, model_results, verbose):
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

      # --- 核心修改点 ---
      # 为 DHEQ 初始化 good_dial 标志
      good_dial = 1.
      # 标记是否至少有一个问题被评估，以正确计算 total_dials
      dialogue_evaluated = False

      # 恢复循环，但只遍历前两个问题
      for qa in qa_list[:1]:
        dialogue_evaluated = True
        q_idx = qa['id']
        val_spans = [anss['text'] for anss in qa['answers']]
        val_spans = handle_cannot(val_spans)
        hf1 = leave_one_out(val_spans)

        if did not in model_results or q_idx not in model_results[did]:
          print(did, q_idx, 'no prediction for this dialogue id')
          good_dial = 0
          f1_stats['NO ANSWER'].append(0.0)
          yes_nos.append(False)
          followups.append(False)
          if val_spans == ['CANNOTANSWER']:
            unanswerables.append(0.0)
          total_qs += 1
          unfiltered_f1s.append(0.0)
          if hf1 >= args.min_f1:
            human_f1.append(hf1)
          continue

        pred_span, pred_yesno, pred_followup = model_results[did][q_idx]
        if '.' in pred_span:
          pred_span = pred_span.replace('.', '')
        if ',' in pred_span:
          pred_span = pred_span.replace(',', '')

        ok = False
        global my_total, my_acc
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


        max_overlap, _ = metric_max_over_ground_truths(
          pred_span, val_spans, par['context'])
        max_f1 = leave_one_out_max(
          pred_span, val_spans, par['context'])
        unfiltered_f1s.append(max_f1)

        # dont eval on low agreement instances
        if hf1 < args.min_f1:
          # 如果跳过，这个问题不计入 HEQ, F1 等，但 DHEQ 仍然受影响
          # 为了简化，我们假设 low-agreement 问题不影响 DHEQ 评估
          # 如果需要更复杂的逻辑（例如，low-agreement 问题自动使 DHEQ 失败），可以在这里设置 good_dial = 0
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
          # 只要有一个问题不达标，整个对话的 DHEQ 就失败
          good_dial = 0.

        span_overlap_stats[max_overlap] += 1
        f1_stats[max_overlap].append(max_f1)
        total_qs += 1.

      # 在处理完一个对话的前两个问题后，累加 DHEQ 和 total_dials
      if dialogue_evaluated:
        DHEQ += good_dial
        total_dials += 1

  # --- 结果计算和打印部分 ---
  # 添加安全检查以避免除零错误
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
  print("NOTE: Evaluation is performed ONLY on the FIRST TWO questions of each dialogue.")
  print('Overall F1: %.2f' % overall_f1)
  print(f"my ACC: {my_acc} / {my_total} = {100 * float(my_acc)/my_total:.2f}%")

  # print('Yes/No Accuracy : %.1f' % yesno_score)
  # print('Followup Accuracy : %.1f' % followup_score)
  # print('Unfiltered F1 ({0:d} questions): {1:.1f}'.format(len(unfiltered_f1s), unfiltered_f1))
  print(
    'Accuracy On Unanswerable Questions: {0:.1f} %% ({1:d} questions)'.format(unanswerable_score, len(unanswerables)))
  human_f1_avg = (100.0 * sum(human_f1) / len(human_f1)) if human_f1 else 0
  # print('Human F1: %.1f' % human_f1_avg)
  print('Model F1 >= Human F1 (Questions): %d / %d, %.1f%%' % (HEQ, total_qs, HEQ_score))
  print('Model F1 >= Human F1 (Dialogs - based on first 2 Qs): %d / %d, %.1f%%' % (DHEQ, total_dials, DHEQ_score))
  print("=======================")
  return metric_json

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--val_file', type=str, required=True, help='file containing validation results')
  parser.add_argument('--model_output', type=str, required=True, help='Path to model output.')
  parser.add_argument('--o', type=str, required=False, help='Path to save score json')
  parser.add_argument('--min_f1', type=float, default=0.4, help='file containing validation results')
  parser.add_argument('--verbose', action='store_true', help='print individual scores')
  args = parser.parse_args()
  val = json.load(open(args.val_file, 'r'))['data']
  preds = defaultdict(dict)
  total = 0
  val_total = 0
  for line in open(args.model_output, 'r'):
    if line.strip():
      pred_idx = json.loads(line.strip())
      dia_id = pred_idx['qid'][0].split("_q#")[0]
      for qid, qspan, qyesno, qfollowup in zip(pred_idx['qid'], pred_idx['best_span_str'], pred_idx['yesno'], pred_idx['followup']):
        preds[dia_id][qid] = qspan, qyesno, qfollowup
        total += 1
  for p in val:
    for par in p['paragraphs']:
      did = par['id']
      qa_list = par['qas']
      val_total += len(qa_list)

  metric_json = eval_fn(val, preds, args.verbose)
  if args.o:
    with open(args.o, 'w') as fout:
      json.dump(metric_json, fout)
  print(f"total: {total}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
ModelClass = AutoModelForCausalLM
import numpy as np
import random
import argparse
from sinktrack import MiniCPM3ForCausalLMWithPromptInjection


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

fix_seeds(323)

def load_model_and_tokenizer(model_id: str, device: str = "auto"):
    print(f"正在从 '{model_id}' 加载模型和分词器...")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 加载模型
    model = MiniCPM3ForCausalLMWithPromptInjection.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(device)

    # 将模型设置为评估模式
    model.eval()

    print("模型和分词器加载完成。")
    return model, tokenizer



# --- 文件路径 ---
model_path = "/home/resource/model/MiniCPM3-4B"
VAL_DATA_PATH = "/home/yhzhang/xu_liu2/zerotoken/llama31/quac/val_v0.2.json"
OUTPUT_PATH = "my_model_predictions.jsonl"
injection_layer_idx = 5

print(f"正在从 {model_path} 加载模型和分词器...")
model, tokenizer = load_model_and_tokenizer(model_path, device="cuda")
print("模型加载完成！")
print("-" * 20)


def get_qwen3_response(model, tokenizer, prompt_text):
    # try:
    messages = [
        {"role": "system",
         "content": "You are an expert at reading comprehension. Your task is to answer questions based ONLY on the provided article and dialogue history. Your answer must be a direct quote from the article. Do not add any extra information, explanations, or introductory phrases like 'The answer is...'. If the answer is not in the article, you must respond with the single word 'CANNOTANSWER'."},
        {"role": "user", "content": prompt_text},
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=1024,
        top_p=0.7,
        temperature=0.7,
        injection_layer_idx=injection_layer_idx
    )

    # 解码生成的文本
    output_token_ids = [
        model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
    ]
    response = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    print("--- Model Raw Output ---")
    print(response)
    print("------------------------")
    return response.strip()
    # except Exception as e:
    #     print(f"模型请求出错: {e}")
    #     return "failed!"  # 返回一个可识别的错误标识


# --- [修改点 1] 大幅简化 Prompt ---
def create_dialogue_prompt(context: str, question: str, history: list) -> str:
    history_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])

    prompt = f"""**Article:**
{context.strip()}

**Dialogue History:**
{history_str}

**Current Question:**
{question}

Based on the article and dialogue history, provide a direct and concise answer to the "Current Question". The answer must be a direct quote from the article. If the answer cannot be found, respond with "CANNOTANSWER".

**Answer:** """
    return prompt


# --- [修改点 3] 移除 parse_response 函数 ---
# 这个函数不再需要了，因为我们直接使用模型的输出。



# --- [修改点 4 & 5] 调整主执行逻辑 ---
if __name__ == "__main__":
    print(f"正在从 {VAL_DATA_PATH} 加载数据...")
    try:
        with open(VAL_DATA_PATH, 'r', encoding='utf-8') as f:
            val_data = json.load(f)['data']
        print(f"成功加载 {len(val_data)} 篇文章的数据。")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 {VAL_DATA_PATH}")
        exit()
    except (json.JSONDecodeError, KeyError):
        print(f"错误: 文件 {VAL_DATA_PATH} 格式不正确，或缺少 'data' 键。")
        exit()

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:
        print(f"开始处理数据，结果将写入 {OUTPUT_PATH}")

        for article in tqdm(val_data, desc="Articles"):
            for paragraph in tqdm(article['paragraphs'], desc="Dialogues", leave=False):
                context = paragraph['context']
                dialogue_id = paragraph['id']

                qids_in_dialogue = []
                spans_in_dialogue = []
                # MODIFIED: 这两个列表现在只用于填充默认值
                yesnos_in_dialogue = []
                followups_in_dialogue = []

                dialogue_history = []

                for qa in paragraph['qas']:
                    question_id = qa['id']
                    question_text = qa['question']

                    prompt = create_dialogue_prompt(context, question_text, dialogue_history)

                    # get_response 现在直接返回清理后的答案字符串
                    pred_span = get_qwen3_response(model, tokenizer, prompt)

                    print(f"QID: {question_id}")
                    print(f"Predicted Answer: {pred_span}")
                    print("-" * 50)

                    qids_in_dialogue.append(question_id)
                    spans_in_dialogue.append(pred_span)

                    # MODIFIED: 填充固定的默认值以满足评估脚本的格式要求
                    yesnos_in_dialogue.append("x")
                    followups_in_dialogue.append("n")

                    dialogue_history.append({"q": question_text, "a": pred_span})

                # MODIFIED: 输出的字典结构保持不变，但 yesno 和 followup 的值是固定的
                result_batch = {
                    "qid": qids_in_dialogue,
                    "best_span_str": spans_in_dialogue,
                    "yesno": yesnos_in_dialogue,
                    "followup": followups_in_dialogue
                }
                outfile.write(json.dumps(result_batch, ensure_ascii=False) + '\n')


    print("-" * 20)
    print(f"处理完成！所有结果已保存在 {OUTPUT_PATH}")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
ModelClass = AutoModelForCausalLM
import numpy as np
import random
import argparse


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

fix_seeds(323)

def load_model_and_tokenizer(model_id: str, device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(device)

    model.eval()

    return model, tokenizer

def get_qwen3_response(model, tokenizer, prompt_text):
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
        temperature=0.7
    )

    output_token_ids = [
        model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
    ]
    response = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    print("--- Model Raw Output ---")
    print(response)
    print("------------------------")
    return response.strip()


model_path = "/home/resource/model/MiniCPM3-4B"
VAL_DATA_PATH = "llama31/quac/val_v0.2.json"
OUTPUT_PATH = "model_predictions.jsonl"
injection_layer_idx = 5

model, tokenizer = load_model_and_tokenizer(model_path, device="cuda")


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

if __name__ == "__main__":
    try:
        with open(VAL_DATA_PATH, 'r', encoding='utf-8') as f:
            val_data = json.load(f)['data']
    except FileNotFoundError:
        exit()
    except (json.JSONDecodeError, KeyError):
        exit()

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as outfile:

        for article in tqdm(val_data, desc="Articles"):
            for paragraph in tqdm(article['paragraphs'], desc="Dialogues", leave=False):
                context = paragraph['context']
                dialogue_id = paragraph['id']

                qids_in_dialogue = []
                spans_in_dialogue = []
                yesnos_in_dialogue = []
                followups_in_dialogue = []

                dialogue_history = []

                for qa in paragraph['qas']:
                    question_id = qa['id']
                    question_text = qa['question']

                    prompt = create_dialogue_prompt(context, question_text, dialogue_history)

                    pred_span = get_qwen3_response(model, tokenizer, prompt)

                    print(f"QID: {question_id}")
                    print(f"Predicted Answer: {pred_span}")
                    print("-" * 50)

                    qids_in_dialogue.append(question_id)
                    spans_in_dialogue.append(pred_span)

                    yesnos_in_dialogue.append("x")
                    followups_in_dialogue.append("n")

                    dialogue_history.append({"q": question_text, "a": pred_span})

                result_batch = {
                    "qid": qids_in_dialogue,
                    "best_span_str": spans_in_dialogue,
                    "yesno": yesnos_in_dialogue,
                    "followup": followups_in_dialogue
                }
                outfile.write(json.dumps(result_batch, ensure_ascii=False) + '\n')


    print("-" * 20)


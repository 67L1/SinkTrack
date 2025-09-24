import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
from sinktrack import LlamaForCausalLMWithPromptInjection
ModelClass = LlamaForCausalLMWithPromptInjection


model_path = "/home/resource/model/Llama-3.1-8B-Instruct"
VAL_DATA_PATH = "val_v0.2.json"
OUTPUT_PATH = "my_model_predictions9.jsonl"
injection_layer_idx = 5

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ModelClass.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)


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


def get_response(prompt_text: str) -> str:
    messages = [
        {"role": "system",
         "content": "You are an expert at reading comprehension. Your task is to answer questions based ONLY on the provided article and dialogue history. Your answer must be a direct quote from the article. Do not add any extra information, explanations, or introductory phrases like 'The answer is...'. If the answer is not in the article, you must respond with the single word 'CANNOTANSWER'."},
        {"role": "user", "content": prompt_text},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=False,
        injection_layer_idx=injection_layer_idx
    )

    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)

    print("--- Model Raw Output ---")
    print(response)
    print("------------------------")

    cleaned_response = response.strip()
    return cleaned_response


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

                    pred_span = get_response(prompt)

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

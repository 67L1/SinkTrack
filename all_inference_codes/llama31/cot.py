import os
import json
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_repo_root() -> str:
    """Repo root = parent of all_inference_codes (works in Colab and locally)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def resolve_path(path: str, repo_root: str) -> str:
    """If path is relative, resolve against repo root; otherwise return as-is."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root, path))


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dialogue_prompt(context: str, question: str, history: list) -> str:
    history_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])
    prompt = f"""**Article:**
{context.strip()}

**Dialogue History:**
{history_str}

**Current Question:**
{question}

Based on the article and dialogue history, provide the reasoning text and a direct and concise answer to the "Current Question". You must provide the final answer in the last line. The answer must be a direct quote from the article.

**Answer:** """
    return prompt


def get_response(prompt_text: str, tokenizer, model, max_new_tokens: int, do_sample: bool) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an expert at reading comprehension. Your task is to answer questions based ONLY on the provided article and dialogue history. Your answer must be a direct quote from the article. If the answer is not in the article, you must respond with the single word 'CANNOTANSWER'."
        },
        {"role": "user", "content": prompt_text},
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()


def run_inference(val_data, tokenizer, model, max_new_tokens, do_sample, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "w", encoding="utf-8") as outfile:
        for article in tqdm(val_data, desc="Articles"):
            for paragraph in tqdm(article["paragraphs"], desc="Dialogues", leave=False):
                context               = paragraph["context"]
                qids_in_dialogue      = []
                spans_in_dialogue     = []
                yesnos_in_dialogue    = []
                followups_in_dialogue = []
                dialogue_history      = []

                for qa in paragraph["qas"]:
                    question_id   = qa["id"]
                    question_text = qa["question"]

                    prompt = create_dialogue_prompt(context, question_text, dialogue_history)

                    try:
                        pred_span = get_response(prompt, tokenizer, model, max_new_tokens, do_sample)
                    except Exception:
                        pred_span = "CANNOTANSWER"

                    if "\n" in pred_span:
                        pred_span = pred_span.split("\n")[-1]
                    if '"' in pred_span:
                        pred_span = pred_span.replace('"', "")

                    qids_in_dialogue.append(question_id)
                    spans_in_dialogue.append(pred_span)
                    yesnos_in_dialogue.append("x")
                    followups_in_dialogue.append("n")
                    dialogue_history.append({"q": question_text, "a": pred_span})

                result_batch = {
                    "qid"          : qids_in_dialogue,
                    "best_span_str": spans_in_dialogue,
                    "yesno"        : yesnos_in_dialogue,
                    "followup"     : followups_in_dialogue,
                }
                outfile.write(json.dumps(result_batch, ensure_ascii=False) + "\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    repo_root = get_repo_root()
    cfg = load_config()

    model_path     = resolve_path(cfg["model"]["model_path"], repo_root)
    torch_dtype    = getattr(torch, cfg["model"]["torch_dtype"])
    device_map     = cfg["model"]["device_map"]
    val_data_path  = resolve_path(cfg["data"]["val_data_path"], repo_root)
    base_dir       = resolve_path(cfg["output"]["base_dir"], repo_root)
    output_stem    = cfg["output"]["cot_output_path"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    do_sample      = cfg["generation"]["do_sample"]
    seeds          = cfg["experiment"]["seeds"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    with open(val_data_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)["data"]

    stem, ext = os.path.splitext(output_stem)
    for seed in seeds:
        set_seed(seed)
        output_path = os.path.join(base_dir, f"{stem}_{seed}{ext}")
        print(f"\n[Seed {seed}] Running inference → {output_path}")
        run_inference(val_data, tokenizer, model, max_new_tokens, do_sample, output_path)
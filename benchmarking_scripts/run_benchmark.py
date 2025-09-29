import random
import numpy as np
import torch

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED=323
fix_seeds(SEED)


from sympy.physics.units import temperature
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import argparse
from ruamel.yaml import YAML
import json
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import json
import copy
import gc
import pickle
import random
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from sinktrack_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithInjection


EVAL_FILE = 'test.json'
DATA_NAME = 'mmstar'

INJECTION_LAYER = 5

model_path = '/home/resource/model/Qwen2.5-VL-7B-Instruct'
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGenerationWithInjection.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained(model_path)
dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]

def get_res(prompt, image, ):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image[0],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=True, injection_layer_idx=INJECTION_LAYER)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]




zero_shot_prompt_template = '''Question: {}
'''

cot_prompt = """Let's think step by step!"""


output_format_options = """
Output Format:
You must give your final answer using the **EXACT format** below:
**Answer: [Your Final Option]**

For example:
[Here is your reasoning text].
**Answer: B**"""

def open_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    output_dir = f'./sinktrack/qwen7b/{DATA_NAME}'
    os.makedirs(output_dir, exist_ok=True)

    mcot_zero_fh = open(output_dir + f'/res_{SEED}.json', 'a')

    for idx, data in enumerate(tqdm(dataset)):
        try:
            print("="*200)
            mcot_input_str = zero_shot_prompt_template.format(data['question'])
            zero_shot_vision = [data['image']]
            zero_shot_mcot_input_str = mcot_input_str + '\n' + cot_prompt + '\n' + output_format_options

            zero_shot = get_res(zero_shot_mcot_input_str, zero_shot_vision)

            zeroshot_mcot_output = copy.deepcopy(data)
            zeroshot_mcot_output['pred'] = zero_shot
            mcot_zero_fh.write(json.dumps(zeroshot_mcot_output) + '\n')
            print(f"zeroshot_mcot_output:\n{zeroshot_mcot_output}\n")

            del zero_shot, zero_shot_vision

            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"eeee:{e}")



if __name__ == '__main__':
    main()
from sympy.physics.units import temperature
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, AutoProcessor
import os
import argparse
from ruamel.yaml import YAML
import json
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import json
import copy
import gc
import pickle
import random
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml', help='global environment configs')
parser.add_argument('--seed', type=int, default=323, help='Random seed for reproducibility.')
args = parser.parse_args()
yaml = YAML()

# Reading a YAML file
with open(args.config, 'r') as file:
    config = yaml.load(file)
    print(config)

########### THE CODE YOU CAN MODIFY  ################
path = 'model/gemma-3-4b-it' # model's path

IMG_FOLDER = '' # image folder with .png
EVAL_FILE = '' # test.json
DATA_NAME = 'realworldqa'

# IMG_FOLDER = '' # image folder with .png
# EVAL_FILE = '' # test.json
# DATA_NAME = 'mmstar'

# IMG_FOLDER = '' # image folder with .png
# EVAL_FILE = '' # test.json
# DATA_NAME = 'POPE'

# IMG_FOLDER = '' # image folder with .png
# EVAL_FILE = '' # test.json
# DATA_NAME = 'm3cot'

######################################################

model = Gemma3ForConditionalGeneration.from_pretrained(
    path, torch_dtype="auto", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(path)


dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]
if DATA_NAME == 'm3cot':
    dataset = [x for x in dataset if x['image'] is not None]

def get_res(prompt, image, one_shot):
    if one_shot:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image[0],
                    },
                    {"type": "text", "text": mcot_induct},
                    {
                        "type": "image",
                        "image": image[1],
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
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
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=2048, temperature=1.5, do_sample=True)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
    return decoded





cot_prompt = """Let's think step by step!"""



zero_shot_prompt_template = '''Question: {}
'''


output_format_options = """
Output Format:
You must give your final answer using the **EXACT format** below:
**Answer: [Your Final Option]**

For example:
[Here is your reasoning text].
**Answer: B**"""

output_format_wo_options = """
Output Format:
You must give your final answer** using the **EXACT format** below:
**Answer: [Your Final Answer Yes or No]**

For example:
[Here is your reasoning text].
**Answer: Yes**"""

def open_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main(SEED):
    import random
    import numpy as np
    import torch

    def fix_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    fix_seeds(SEED)
    output_dir = './cot/qwen4b/{}'.format(DATA_NAME)
    os.makedirs(output_dir, exist_ok=True)

    mcot_zero_fh = open(output_dir + f'/res_{SEED}.json', 'a')

    for idx, data in enumerate(tqdm(dataset)):
        try:
            print("="*200)
            mcot_input_str = zero_shot_prompt_template.format(data['question'])
            if DATA_NAME == 'm3cot':
                for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F'], data['choices']):
                    mcot_input_str += '{}. {}\n'.format(i, c)


            zero_shot_vision = [os.path.join(IMG_FOLDER, data['image'])]


            if DATA_NAME == 'POPE':
                zero_shot_mcot_input_str = mcot_input_str+ '\n' + cot_prompt + '\n' + output_format_wo_options
            else:
                zero_shot_mcot_input_str = mcot_input_str+ '\n' + cot_prompt + '\n' + output_format_options



            zero_shot = get_res(zero_shot_mcot_input_str, zero_shot_vision, one_shot=False)


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

    main(SEED=args.seed)
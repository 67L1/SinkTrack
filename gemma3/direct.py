from sympy.physics.units import temperature
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, AutoProcessor
import os
import argparse
from ruamel.yaml import YAML
import json
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
parser.add_argument('--config', default='config/config.yaml', help='global environment configs')
args = parser.parse_args()
yaml = YAML()

# Reading a YAML file
with open(args.config, 'r') as file:
    config = yaml.load(file)
    print(config)

IMG_FOLDER = 'MMStar/'
EVAL_FILE = 'MMStar/test.json'
DATA_NAME = 'mmstar'


path = "/home/resource/model/gemma-3-4b-it"
# default: Load the model on the available device(s)
model = Gemma3ForConditionalGeneration.from_pretrained(
    path, torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained(path)


dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]

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
        generation = model.generate(**inputs, max_new_tokens=64, temperature=2.0, top_p=0.9, do_sample=True)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
    return decoded





cot_prompt = """Let's think step by step!"""



zero_shot_prompt_template = '''Question: {}
'''


output_format_options = """
Output Format:
You must give your final answer using the **EXACT format** below **directly**:
**Answer: [Your Final Option]**

For example:
**Answer: B**"""

output_format_wo_options = """
Output Format:
You must give your final answer using the **EXACT format** below **directly**:
**Answer: [Your Final Answer]**

For example:
**Answer: Yes**"""

def open_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    if '4b' in path:
        output_dir = './results/qwen4b/{}'.format(DATA_NAME)
    else:
        output_dir = './results/qwen12b/{}'.format(DATA_NAME)
    os.makedirs(output_dir, exist_ok=True)

    mcot_zero_fh = open(output_dir + '/res.json', 'a')

    for idx, data in enumerate(tqdm(dataset)):
        print("="*200)
        final_output_format = ""
        mcot_input_str = zero_shot_prompt_template.format(data['question'])

        if final_output_format == "":
            final_output_format = output_format_options
        if DATA_NAME == 'POPE':
            final_output_format = output_format_wo_options



        zero_shot_vision = [os.path.join(IMG_FOLDER, data['image'])]
        zero_shot_mcot_input_str = mcot_input_str+ '\n' + final_output_format
        zero_shot = get_res(zero_shot_mcot_input_str, zero_shot_vision, one_shot=False)
        zeroshot_mcot_output = copy.deepcopy(data)
        zeroshot_mcot_output['pred'] = zero_shot
        mcot_zero_fh.write(json.dumps(zeroshot_mcot_output) + '\n')
        print(f"zeroshot_mcot_output:\n{zeroshot_mcot_output}\n")

        del zero_shot, zero_shot_vision

        torch.cuda.empty_cache()
        gc.collect()



if __name__ == '__main__':

    main()

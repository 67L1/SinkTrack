import pandas as pd
from PIL import Image
import io
import os
import json
from tqdm import tqdm
import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj


parquet_file_paths = [
    'M3CoT/data/test-00000-of-00002.parquet',
    'M3CoT/data/test-00001-of-00002.parquet'
]

output_json_path = 'M3CoT/test.json'
output_image_folder = 'M3CoT/images'

os.makedirs(output_image_folder, exist_ok=True)

all_dataframes = []

for file_path in parquet_file_paths:
    if os.path.exists(file_path):
        try:
            df_part = pd.read_parquet(file_path)
            all_dataframes.append(df_part)
            print(f"loaded {file_path}")
        except Exception as e:
            print(f"{e}")
            exit()
    else:
        print(f"Do not exist: {file_path}")
        exit()


df_merged = pd.concat(all_dataframes, ignore_index=True)


with open(output_json_path, 'w', encoding='utf-8') as json_file:
    for index, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Processing"):
        raw_image_data = row.get('image')
        if raw_image_data is None:
            continue
        image_bytes = None
        if isinstance(raw_image_data, dict) and 'bytes' in raw_image_data:
            image_bytes = raw_image_data['bytes']
        elif isinstance(raw_image_data, bytes):
            image_bytes = raw_image_data

        if not image_bytes:
            continue

        question_id = row['id']

        try:
            image_stream = io.BytesIO(image_bytes)
            img = Image.open(image_stream)

            image_filename = f"{question_id}.png"
            save_path = os.path.join(output_image_folder, image_filename)
            img.save(save_path, 'png')

            choices_data = convert_numpy_types(row['choices'])
            answer_data = convert_numpy_types(row['answer'])
            id_data = convert_numpy_types(question_id)

            record = {
                'id': id_data,
                'image': os.path.join(output_image_folder, image_filename),
                'question': row['question'],
                'choices': choices_data,
                'answer': answer_data
            }

            json_file.write(json.dumps(record, ensure_ascii=False) + '\n')

        except Exception as e:
            print(f"{e}")
            continue

print(f"Done, saved to: {output_json_path}")
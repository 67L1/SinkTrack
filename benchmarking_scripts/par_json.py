import pandas as pd
from PIL import Image
import io
import os
import json
from tqdm import tqdm

parquet_file_paths = ['mmstar.parquet']

output_json_path = 'test.json'
output_image_folder = 'images'

os.makedirs(output_image_folder, exist_ok=True)

all_dataframes = []

for file_path in parquet_file_paths:
    try:
        df_part = pd.read_parquet(file_path)
        all_dataframes.append(df_part)
    except FileNotFoundError:
        exit()
    except Exception as e:
        exit()

if not all_dataframes:
    exit()

df_merged = pd.concat(all_dataframes, ignore_index=True)

print(df_merged.head())

with open(output_json_path, 'w', encoding='utf-8') as json_file:
    for index, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="processing..."):
        unique_id = index

        record = {
            'id': unique_id,
            'question': row['question'],
            'answer': row['answer'],
            'image': None
        }

        if 'image' in row and row['image'] is not None:
            image_bytes = row['image']

            try:
                image_stream = io.BytesIO(image_bytes)
                img = Image.open(image_stream)
                image_filename = f"{unique_id}.png"
                image_filepath = os.path.join(output_image_folder, image_filename)
                img.save(image_filepath, 'PNG')
                record['image'] = os.path.join(os.path.basename(output_image_folder), image_filename)
            except Exception as e:
                print(f"{e}")

        json_file.write(json.dumps(record, ensure_ascii=False) + '\n')

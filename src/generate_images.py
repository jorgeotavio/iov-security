import numpy as np
from PIL import Image
import os
import re
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def hex_to_binary(hex_data):
    binary_data = bin(int(hex_data, 16))[2:].zfill(len(hex_data)*4)
    return [int(b) for b in binary_data]

def convert_message_to_byteplot(message, output_folder, file_count):
    hex_data = message.replace(' ', '')
    binary_data = hex_to_binary(hex_data)

    byte_image = np.array(binary_data, dtype=np.uint8).reshape(-1, 1)
    size = int(np.ceil(np.sqrt(len(byte_image))))
    if size * size < len(byte_image):
        size += 1

    byte_image = np.pad(byte_image, ((0, size * size - len(byte_image)), (0, 0)), 'constant')

    gray_image = Image.fromarray(byte_image * 255, 'L').resize((size, size))
    rgb_image = gray_image.convert('RGB')
    resized_image = rgb_image.resize((224, 224), Image.Resampling.LANCZOS)

    image_save_path = os.path.join(output_folder, f'{file_count}.png')
    resized_image.save(image_save_path)

def split_dataset(dataset_path):
    with open(dataset_path, 'r') as file:
        regex = r'DLC: \d+    ([0-9a-fA-F ]+)'
        lines = [
            re.search(regex, line).group(1) for line in file.readlines() if re.search(regex, line)
        ]

    random.shuffle(lines)
    size_of_dataset = int(len(lines) * 0.1)
    lines = lines[:size_of_dataset]

    split_index = int(len(lines) * 0.7)
    return lines[:split_index], lines[split_index:]

def process_lines(lines, train_folder, progress_bar):
    for i, line in enumerate(lines):
        convert_message_to_byteplot(line, train_folder, i)
        progress_bar.update(1)

def process_dataset(dataset_path, train_folder, validation_folder, progress_bar):
    train_lines, val_lines = split_dataset(dataset_path)

    print("Dataset: ", dataset_path)
    print("Total Train Lines: ", len(train_lines))
    print("Total Validate Lines: ", len(val_lines))

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_lines, train_lines, train_folder, tqdm(total=len(train_lines), desc="Processing train lines")),
                   executor.submit(process_lines, val_lines, validation_folder, tqdm(total=len(val_lines), desc="Processing validation lines"))]
        for future in as_completed(futures):
            future.result()

def process_all_datasets(datasets_folder, output_base_folder):
    with ProcessPoolExecutor() as executor:
        for filename in os.listdir(datasets_folder):
            if filename.endswith('.txt'):
                dataset_path = os.path.join(datasets_folder, filename)
                dataset_name = os.path.splitext(filename)[0]
                train_folder = os.path.join(output_base_folder, 'train', dataset_name)
                validation_folder = os.path.join(output_base_folder, 'validation', dataset_name)

                executor.submit(process_dataset, dataset_path, train_folder, validation_folder, tqdm(desc=f"Processing dataset: {dataset_name}", position=0, leave=False))

print("Working with this total of CPUs: ", os.cpu_count())

path = Path(os.getcwd())
datasets_folder = str(path) + "/data/datasets"
output_base_folder = str(path) + "/data"

process_all_datasets(datasets_folder, output_base_folder)

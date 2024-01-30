import numpy as np
from PIL import Image
import os
import re
from pathlib import Path
import random

def hex_to_binary(hex_data):
    binary_data = bin(int(hex_data, 16))[2:].zfill(len(hex_data)*4)
    return [int(b) for b in binary_data]

def convert_message_to_byteplot(message, output_folder, file_count):
    match = re.search(r'DLC: \d+    ([0-9a-fA-F ]+)', message)
    if match:
        hex_data = match.group(1).replace(' ', '')
        binary_data = hex_to_binary(hex_data)

        byte_image = np.array(binary_data, dtype=np.uint8).reshape(-1, 1)
        size = int(np.ceil(np.sqrt(len(byte_image))))
        if size*size < len(byte_image):
            size += 1

        byte_image = np.pad(byte_image, ((0, size*size - len(byte_image)), (0, 0)), 'constant')

        gray_image = Image.fromarray(byte_image * 255, 'L').resize((size, size))
        rgb_image = gray_image.convert('RGB')
        resized_image = rgb_image.resize((224, 224), Image.Resampling.LANCZOS)

        image_save_path = os.path.join(output_folder, f'{file_count}.png')
        resized_image.save(image_save_path)

def split_dataset(dataset_path, train_ratio=0.7):
    with open(dataset_path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)
    split_index = int(len(lines) * train_ratio)
    return lines[:split_index], lines[split_index:]

def process_dataset(dataset_path, train_folder, validation_folder):
    train_lines, val_lines = split_dataset(dataset_path)

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    for i, line in enumerate(train_lines):
        convert_message_to_byteplot(line, train_folder, i)

    for i, line in enumerate(val_lines):
        convert_message_to_byteplot(line, validation_folder, i)

def process_all_datasets(datasets_folder, output_base_folder):
    for filename in os.listdir(datasets_folder):
        if filename.endswith('.txt'):
            dataset_path = os.path.join(datasets_folder, filename)
            dataset_name = os.path.splitext(filename)[0]

            train_folder = os.path.join(output_base_folder, 'train', dataset_name)
            validation_folder = os.path.join(output_base_folder, 'validation', dataset_name)

            process_dataset(dataset_path, train_folder, validation_folder)

# Caminho para a pasta com os datasets e a pasta base para as imagens
path = Path(os.getcwd())
datasets_folder = str(path) + "/data/datasets"
output_base_folder = str(path) + "/data"

process_all_datasets(datasets_folder, output_base_folder)

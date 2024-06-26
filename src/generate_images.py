import os
import sys
import re
import random
import multiprocessing
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from utils import is_dev
import time

folders_config = {
    'output_train_folder': 'data/dev/images_train/',
    'output_validation_folder': 'data/dev/images_validation/',
    'train_ratio': 0.7
} if is_dev() else {
    'output_train_folder': 'data/prod/images_train/',
    'output_validation_folder': 'data/prod/images_validation/',
    'train_ratio': 0.7
}

datasets = [
    {
        'name': 'attack_free',
        'dataset': '/data/prod/datasets/attack_free.txt',
        'dataset_dev': '/data/dev/datasets/attack_free.txt',
    },
    {
        'name': 'dos',
        'dataset': '/data/prod/datasets/dos.txt',
        'dataset_dev': '/data/dev/datasets/dos.txt',
    },
    {
        'name': 'fuzzy',
        'dataset': '/data/prod/datasets/fuzzy.txt',
        'dataset_dev': '/data/dev/datasets/fuzzy.txt',
    },
]

def hex_to_binary(hex_data):
    binary_data = bin(int(hex_data, 16))[2:].zfill(len(hex_data)*4)
    return [int(b) for b in binary_data]

def get_hex_data_from_line(line):
    hex_data = re.search(r'DLC: \d+    ([0-9a-fA-F ]+)', line)
    if hex_data:
        return hex_data.group(1).replace(' ', '')

def get_lines(file_path):
    path = str(Path(os.getcwd()))
    with open(path + file_path, 'r') as file:
        return file.readlines()

def convert_bin_to_image(binary_string, path_to_save, file_count):
    byte_data = np.array(binary_string, dtype=np.uint8)
    length = len(byte_data)
    
    size = int(np.ceil(np.sqrt(length)))
    padded_length = size * size
    
    if length < padded_length:
        byte_data = np.pad(byte_data, (0, padded_length - length), 'constant')
    
    byte_image = byte_data.reshape((size, size))
    
    gray_image = Image.fromarray(byte_image * 255, 'L')
    rgb_image = gray_image.convert('RGB')
    resized_image = rgb_image.resize((224, 224), Image.Resampling.NEAREST)

    image_save_path = os.path.join(path_to_save, f'img-{file_count}.png')
    resized_image.save(image_save_path)

def generate(lines, path_to_save, bar_position):
    for index, line in enumerate(tqdm(lines, position=bar_position, desc=f'Bar {bar_position}')):
        hex_data = get_hex_data_from_line(line)
        if hex_data and hex_data != '0000000000000000':
            binary_data = hex_to_binary(hex_data)
            convert_bin_to_image(binary_data, path_to_save, index)

def start():
    process_list = []
    args_list = []
    for dataset in datasets:
        dataset_type = 'dataset_dev' if is_dev() else 'dataset'
        lines = get_lines(dataset[dataset_type])
        random.shuffle(lines)
        split_index = int(len(lines) * folders_config['train_ratio'])

        train_lines = lines[:split_index]
        validation_lines = lines[split_index:]

        path_to_save_train = os.path.join(folders_config['output_train_folder'], dataset['name'])
        path_to_save_validation = os.path.join(folders_config['output_validation_folder'], dataset['name'])
        args_list.append(({ 'lines':  train_lines, 'path_to_save': path_to_save_train}))
        args_list.append(({ 'lines':  validation_lines, 'path_to_save': path_to_save_validation}))

    for i, args in enumerate(args_list):
        args['bar_position'] = i+1
        process = multiprocessing.Process(target=generate, kwargs=args)
        process_list.append(process)
    
    for process in process_list:
        process.start()
        time.sleep(1)
    
    for process in process_list:
        process.join()

if __name__ == '__main__':
    print('start flow from src/main.py')

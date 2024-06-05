import os
import sys
import re
import random
import multiprocessing
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# - get files
# - get line
# - convert the line
# - save image generated

folders_config = {
    'output_train_folder': 'data/images_train/',
    'output_validation_folder': 'data/images_validation/',
    'train_ratio': 0.7
}

datasets = [
    {
        'name': 'attack_free',
        'dataset': '/data/datasets/attack_free.txt',
        'dataset_dev': '/data/datasets_dev/attack_free.dev.txt',
    },
    {
        'name': 'dos',
        'dataset': '/data/datasets/dos.txt',
        'dataset_dev': '/data/datasets_dev/dos.dev.txt',
    },
    {
        'name': 'fuzzy',
        'dataset': '/data/datasets/fuzzy.txt',
        'dataset_dev': '/data/datasets_dev/fuzzy.dev.txt',
    },
]

def is_dev():
    args = sys.argv[1:]
    if (len(args) > 1):
        return sys.argv[1] == 'dev'
    return False

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
    byte_image = np.array(binary_string, dtype=np.uint8).reshape(-1, 1)
    size = int(np.ceil(np.sqrt(len(byte_image))))

    if size*size < len(byte_image):
        size += 1

    byte_image = np.pad(byte_image, ((0, size*size - len(byte_image)), (0, 0)), 'constant')

    gray_image = Image.fromarray(byte_image * 255, 'L').resize((size, size))
    rgb_image = gray_image.convert('RGB')
    resized_image = rgb_image.resize((224, 224), Image.Resampling.LANCZOS)

    image_save_path = os.path.join(path_to_save, f'img-{file_count}.png')
    resized_image.save(image_save_path)

def generate(lines, path_to_save, datased_id):
    for index, line in enumerate(tqdm(lines, position=datased_id)):
        hex = get_hex_data_from_line(line)
        if hex:
            bin = hex_to_binary(hex)
            convert_bin_to_image(bin, path_to_save, index)

def worker(lines_chunk, path_to_save, worker_id):
    generate(lines_chunk, path_to_save, worker_id)

def start():
    num_cores = multiprocessing.cpu_count()
    process_list = []

    for index, dataset in enumerate(datasets):
        dataset_type = 'dataset_dev' if is_dev() else 'dataset'
        lines = get_lines(dataset[dataset_type])
        random.shuffle(lines)
        split_index = int(len(lines) * folders_config['train_ratio'])

        train_lines = lines[:split_index]
        validation_lines = lines[split_index:]

        path_to_save_train = os.path.join(folders_config['output_train_folder'], dataset['name'])
        path_to_save_validation = os.path.join(folders_config['output_validation_folder'], dataset['name'])

        train_chunks = np.array_split(train_lines, num_cores)
        validation_chunks = np.array_split(validation_lines, num_cores)

        for i in range(num_cores):
            train_process = multiprocessing.Process(target=worker, args=(train_chunks[i], path_to_save_train, index * num_cores + i))
            validation_process = multiprocessing.Process(target=worker, args=(validation_chunks[i], path_to_save_validation, (index * num_cores + i) + num_cores))
            process_list.append(train_process)
            process_list.append(validation_process)

    for proc in process_list:
        proc.start()

    for proc in process_list:
        proc.join()

if __name__ == '__main__':
    start()

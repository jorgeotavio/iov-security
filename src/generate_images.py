import os
import re
import multiprocessing
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import is_dev
import time
from pathlib import Path

folders_config = {
    'output_train_folder': 'data/dev/images_train/',
    'output_validation_folder': 'data/dev/images_validation/',
} if is_dev() else {
    'output_train_folder': 'data/prod/images_train/',
    'output_validation_folder': 'data/prod/images_validation/',
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
        for line in file:
            yield line

def convert_bin_to_image(binary_string, path_to_save, index_line):
    byte_data = np.array(binary_string, dtype=np.uint8)
    length = len(byte_data)

    size = int(np.ceil(np.sqrt(length)))
    padded_length = size * size

    if length < padded_length:
        byte_data = np.pad(byte_data, (0, padded_length - length), 'constant')

    byte_image = byte_data.reshape((size, size))

    gray_image = Image.fromarray(byte_image * 255, 'L')
    rgb_image = gray_image.convert('RGB')
    resized_image = rgb_image.resize((28, 28), Image.Resampling.NEAREST)

    image_save_path = os.path.join(path_to_save, f'img-{index_line}.png')
    resized_image.save(image_save_path)

def generate(lines, path_to_save, progress):
    os.makedirs(path_to_save, exist_ok=True)
    existing_files = set(os.listdir(path_to_save))

    for i, line in enumerate(lines):
        image_name = f'img-{i}.png'
        if image_name in existing_files:
            progress.value += 1
            continue

        hex_data = get_hex_data_from_line(line)
        if hex_data:
            binary_data = hex_to_binary(hex_data)
            convert_bin_to_image(binary_data, path_to_save, i)

        progress.value += 1

def start():
    process_list = []
    args_list = []
    total_lines = 0

    for dataset in datasets:
        total_lines += int(len(list(get_lines(dataset['dataset']))))

    with multiprocessing.Manager() as manager:
        progress = manager.Value('i', 0)

        with tqdm(total=total_lines) as pbar:

          for dataset in datasets:
              # lines = get_lines(dataset['dataset'])
              # random.shuffle(lines)
              split_index = int(len(list(enumerate(get_lines(dataset['dataset'])))) * 0.7)

              train_lines = []
              validation_lines = []

              for i, line in enumerate(get_lines(dataset['dataset'])):
                  if i < split_index:
                      train_lines.append(line)
                  else:
                      validation_lines.append(line)

              path_to_save_train = os.path.join(folders_config['output_train_folder'], dataset['name'])
              path_to_save_validation = os.path.join(folders_config['output_validation_folder'], dataset['name'])

              args_list.append(({ 'lines':  train_lines, 'path_to_save': path_to_save_train }))
              args_list.append(({ 'lines':  validation_lines, 'path_to_save': path_to_save_validation }))

          print('1 - criando a lista de processos')
          for i, args in enumerate(args_list):
              args['progress'] = progress
              process = multiprocessing.Process(target=generate, kwargs=args)
              process_list.append(process)

          print('2 - iniciando os processos')
          for process in process_list:
              process.start()
              time.sleep(1)

          print('3 - iniciando progress bar')
          while any(r.is_alive() for r in process_list):
              pbar.n = progress.value
              pbar.refresh()
              time.sleep(0.1)

          print('3 - finalizando os processos')
          for process in process_list:
              process.join()

          print('4 - finalizando progress bar')
          pbar.close()

if __name__ == '__main__':
    start()
    print('start flow from src/main.py')

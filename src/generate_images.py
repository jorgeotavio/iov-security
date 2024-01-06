import numpy as np
from PIL import Image
from pathlib import Path
import os
import re

def convert_can_log_to_byteplot(dataset_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    extensions = ['txt']

    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        all_bytes = []

        extension = filename.split('.')[-1]

        if extension not in extensions:
            continue

        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(r'DLC: \d+    ([0-9a-fA-F ]+)', line)
                if match:
                    hex_data = match.group(1).replace(' ', '')
                    bytes_data = bytes.fromhex(hex_data)
                    all_bytes.extend(bytes_data)

        byte_array = np.array(all_bytes, dtype=np.uint8)

        size = int(np.ceil(np.sqrt(len(byte_array))))
        if size*size < len(byte_array):
            size = size + 1

        byte_image = np.zeros((size*size,), dtype=np.uint8)
        byte_image[:len(byte_array)] = byte_array

        byte_image = np.reshape(byte_image, (size, size))

        image = Image.fromarray(byte_image, 'L')
        image.save(os.path.join(output_folder, f'{filename}.png'))

path = Path(os.getcwd())
data_dir = str(path) + "/data/"

convert_can_log_to_byteplot(data_dir + 'datasets', data_dir + 'images')

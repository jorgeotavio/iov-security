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
            size += 1

        byte_image = np.zeros((size*size,), dtype=np.uint8)
        byte_image[:len(byte_array)] = byte_array
        byte_image = np.reshape(byte_image, (size, size))

        gray_image = Image.fromarray(byte_image, 'L')
        rgb_image = gray_image.convert('RGB')
        resized_image = rgb_image.resize((224, 224), Image.Resampling.LANCZOS)

        class_name = filename.split('_dataset')[0]

        class_dir = os.path.join(output_folder, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        image_save_path = os.path.join(class_dir, f'img.png')
        resized_image.save(image_save_path)

path = Path(os.getcwd())
data_dir = str(path) + "/data/"

convert_can_log_to_byteplot(data_dir + 'datasets', data_dir + 'images')

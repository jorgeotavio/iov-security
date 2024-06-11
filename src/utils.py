import sys
import glob
from tqdm import tqdm
import os

def is_dev():
    args = sys.argv[1:]

    if (len(args) > 2):
        return sys.argv[3] == 'dev'
    
    if (len(args) > 1):
        return sys.argv[2] == 'dev'
    return False

def delete_images():
    root_folder = 'data/dev' if is_dev() else 'data/prod'
    png_files = glob.glob(os.path.join(root_folder, '**', '*.png'), recursive=True)
    
    for file_path in tqdm(png_files, desc="Deleting images", unit="file"):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error to delete {file_path}: {e}")

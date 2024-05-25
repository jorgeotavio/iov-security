import os
from pathlib import Path

path_base = str(Path(os.getcwd()))

folders_config = {
    'train_dir': path_base + '/data/images_train',
    'val_dir': path_base + '/data/images_validation',
    'result_dir': path_base + '/data/train_result'
}

pconfig = {
    'num_classes': 3,
    'num_epochs': 4,
    'use_gpu': True,
    'use_data_percentage': 80/100,
    'batch_size': 32
}
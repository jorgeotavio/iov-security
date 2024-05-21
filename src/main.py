import sys
from generate_images import start as start_generate_images
from train_model import start as start_train_model

def generate_images():
    start_generate_images()

def start_train():
    start_train_model()

args = sys.argv[1:]

if(len(args) > 1):
    if args[0] == 'gen':
        generate_images()

    if args[0] == 'train':
        start_train()
import sys
from generate_images import start as start_generate_images
from train import start as start_train
import torch

def generate_images():
    start_generate_images()

if __name__ == '__main__':
    args = sys.argv[1:]

    if(len(args) > 0):
        if args[0] == 'gen_imgs':
            generate_images()

        if args[0] == 'train':
            print('Starting train')
            if args[1]:
                start_train(args[1])
        
        if args[0] == 'test_cuda':
            if torch.cuda.is_available():
                print(torch.cuda.device_count())
                print(torch.cuda.get_device_name(0))
                print("With GPU CUDA.")
            else:
                print("Without GPU.")
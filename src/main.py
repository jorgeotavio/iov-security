import sys
from configs import folders_config, pconfig
from generate_images import start as start_generate_images
# from train_shufflenet import start as start_train_shufflenet
from train_shufflenet_fastai import start as start_train_shufflenet
import torch

def generate_images():
    start_generate_images()

def start_train():
    # start_train_shufflenet(**folders_config, **pconfig)
    start_train_shufflenet()

if __name__ == '__main__':
    args = sys.argv[1:]

    if(len(args) > 0):
        if args[0] == 'gen_imgs':
            generate_images()

        if args[0] == 'train':
            print('Starting train of shufflenet')
            start_train()
        
        if args[0] == 'test_cuda':
            if torch.cuda.is_available():
                print(torch.cuda.device_count())
                print(torch.cuda.get_device_name(0))
                print("With GPU CUDA.")
            else:
                print("Without GPU.")
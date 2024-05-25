import sys
from configs import folders_config, pconfig
from generate_images import start as start_generate_images
# from train_shufflenet import start as start_train_shufflenet
from train_shufflenet_fastai import start as start_train_shufflenet

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
            print('starting train of shufflenet')
            start_train()
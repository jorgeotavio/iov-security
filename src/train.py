from fastai.vision.all import *
from pathlib import Path
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights, vgg16_bn, VGG16_BN_Weights
import torch
import os
from utils import is_dev

# mp.set_start_method('spawn', force=True)

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

path_train = Path('data/dev/images_train' if is_dev() else 'data/prod/images_train')
path_valid = Path('data/dev/images_validation' if is_dev() else 'data/prod/images_validation')

def get_learner(arch, dls):
    architectures = {
        'vgg16' : {
            'model': vgg16_bn,
            'weight': VGG16_BN_Weights.DEFAULT
        },
        'shuffleNetV2' : {
            'model': shufflenet_v2_x1_0,
            'weight': ShuffleNet_V2_X1_0_Weights.DEFAULT
        },
    }

    if arch not in architectures:
        raise ValueError(f"Arquitetura {arch} não é suportada. Escolha uma das seguintes: {list(architectures.keys())}")

    print(f'Trainning {arch}')

    model = architectures[arch]['model']
    weight = architectures[arch]['weight']
    learner = vision_learner(dls, model, weights=weight, metrics=accuracy)
    return learner

def train_with_progress(learner, epochs, lr, cbs=None, arch=''):
    cbs = cbs or []
    for epoch in range(epochs):
        learner.fit_one_cycle(1, lr, cbs=cbs)
        learner.save(f'{arch}/model_epoch_{epoch+1}')

def start(arch):
    num_workers = os.cpu_count()

    dls = ImageDataLoaders.from_folder(
        path_train, 
        valid_pct=0.3, 
        seed=42, 
        item_tfms=Resize(224), 
        batch_tfms=aug_transforms(), 
        valid_folder=path_valid,
        num_workers=num_workers,
        batch_size=512,
    )

    learner = get_learner(arch, dls)

    if torch.cuda.is_available():
        print("Using GPU CUDA.")
        learner.to(torch.device('cuda'))

    elif torch.backends.mps.is_available():
        print("Using MPS device.")
        learner.to(torch.device('mps'))

    else:
        print("Using CPU.")
        learner.to(torch.device('cpu'))

    epochs = 5
    lr = 1e-3

    checkpoint_callback = SaveModelCallback(monitor='accuracy', fname='best_model')
    train_with_progress(learner, epochs, lr, cbs=[checkpoint_callback], arch=arch)

    learner.load('best_model')

if __name__ == '__main__':
    print('start flow from src/main.py')
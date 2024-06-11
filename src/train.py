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
        raise ValueError(f"Archteture {arch} is not supported. choose one of the follow options: {list(architectures.keys())}")

    print(f'Treinando {arch}')

    model = architectures[arch]['model']
    weight = architectures[arch]['weight']
    learner = vision_learner(dls, model, weights=weight, metrics=accuracy)
    return learner

def train_with_progress(learner, epochs, lr, cbs=None, arch=''):
    cbs = cbs or []
    for epoch in range(epochs):
        learner.fit_one_cycle(1, lr, cbs=cbs)
        learner.save(f'{arch}/model_epoch_{epoch+1}', with_opt=True)

def start(arch):
    print('---- DEV MODE ----' if is_dev() else '---- STARTING TRAIN ----')
    num_workers = os.cpu_count()

    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Mean and std for normalization

    dls = ImageDataLoaders.from_folder(
        path_train, 
        valid_pct=0.3, 
        seed=42, 
        # item_tfms=Resize(224), 
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)], 
        valid_folder=path_valid,
        num_workers=num_workers,
        batch_size=16,
    )

    learner = get_learner(arch, dls)

    if torch.cuda.is_available():
        print("Using GPU CUDA.")
        learner.to_fp16()
        learner.to(torch.device('cuda'))

    elif torch.backends.mps.is_available():
        print("Using MPS device.")
        learner.to(torch.device('mps'))

    else:
        print("Using CPU.")
        learner.to(torch.device('cpu'))

    epochs = 5
    lr = 1e-3

    checkpoint_callback = SaveModelCallback(monitor='accuracy', fname=f'{arch}_best_model', with_opt=True)
    train_with_progress(learner, epochs, lr, cbs=[checkpoint_callback], arch=arch)

    learner.load(f'{arch}_best_model', with_opt=True)

if __name__ == '__main__':
    print('start flow from src/main.py')

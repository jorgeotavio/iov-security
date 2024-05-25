from fastai.vision.all import *
from pathlib import Path
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
import torch
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Define os caminhos para as pastas de dados
path_train = Path('data/images_train')
path_valid = Path('data/images_validation')
path_result = Path('data/train_result')

def get_learner(arch, dls, pretrained=True):
    architectures = {
        # 'resnet34': resnet34,
        # 'resnet50': resnet50,
        # 'vgg16': vgg16_bn,
        # 'vgg19': vgg19_bn,
        # 'mobilenet_v2': mobilenet_v2,
        'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    }

    if arch not in architectures:
        raise ValueError(f"Arquitetura {arch} não é suportada. Escolha uma das seguintes: {list(architectures.keys())}")

    model = architectures[arch]
    learner = vision_learner(dls, model, weights=ShuffleNet_V2_X1_0_Weights.DEFAULT, metrics=accuracy)
    return learner

def train_with_progress(learner, epochs, lr, cbs=None):
    cbs = cbs or []
    for epoch in range(epochs):
        learner.fit_one_cycle(1, lr, cbs=cbs)
        learner.save(f'{path_result}/model_epoch_{epoch+1}')

def start():
    dls = ImageDataLoaders.from_folder(
        path_train, 
        valid_pct=0.2, 
        seed=42, 
        item_tfms=Resize(224), 
        batch_tfms=aug_transforms(), 
        valid_folder=path_valid
    )

    arch = 'shufflenet_v2_x1_0'
    learner = get_learner(arch, dls)

    if torch.backends.mps.is_available():
        print("Usando dispositivo MPS.")
        learner.to(torch.device('mps'))
    else:
        print("Dispositivo MPS não disponível, usando CPU.")
        learner.to(torch.device('cpu'))

    epochs = 5
    lr = 1e-3

    checkpoint_callback = SaveModelCallback(monitor='accuracy', fname='best_model')
    train_with_progress(learner, epochs, lr, cbs=[checkpoint_callback])

    learner.load('best_model')

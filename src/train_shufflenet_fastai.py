from fastai.vision.all import *
from pathlib import Path
from tqdm.notebook import tqdm
from torchvision.models import shufflenet_v2_x1_0
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Define os caminhos para as pastas de dados
path_train = Path('data/images_train')
path_valid = Path('data/images_validation')
path_result = Path('data/train_result')

def get_learner(arch, dls, pretrained=True):
    """
    Função para criar um Learner com a arquitetura especificada.

    Args:
    arch (str): Nome da arquitetura. Ex: 'resnet34', 'vgg16', 'mobilenet_v2', etc.
    dls (DataLoaders): DataLoaders do FastAI.
    pretrained (bool): Se True, usa pesos pré-treinados no ImageNet.

    Returns:
    Learner: Objeto Learner do FastAI.
    """
    # Dicionário de arquiteturas disponíveis
    architectures = {
        # 'resnet34': resnet34,
        # 'resnet50': resnet50,
        # 'vgg16': vgg16_bn,
        # 'vgg19': vgg19_bn,
        # 'mobilenet_v2': mobilenet_v2,
        'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
        # Adicione mais arquiteturas conforme necessário
    }

    # Verifica se a arquitetura fornecida está disponível
    if arch not in architectures:
        raise ValueError(f"Arquitetura {arch} não é suportada. Escolha uma das seguintes: {list(architectures.keys())}")

    # Cria um Learner com a arquitetura especificada
    model = architectures[arch]
    learner = cnn_learner(dls, model, pretrained=pretrained, metrics=accuracy)
    return learner

# Função para treinamento com barra de progresso
def train_with_progress(learner, epochs, lr, cbs=None):
    """
    Treina o modelo com uma barra de progresso.

    Args:
    learner (Learner): O objeto Learner do FastAI.
    epochs (int): Número de épocas para treinar.
    lr (float): Taxa de aprendizado.
    cbs (list): Lista de callbacks adicionais (opcional).
    """
    cbs = cbs or []
    for epoch in tqdm(range(epochs), desc="Treinando"):
        learner.fit_one_cycle(1, lr, cbs=cbs)
        learner.save(f'{path_result}/model_epoch_{epoch+1}')

def start():

    # Configura os DataLoaders
    dls = ImageDataLoaders.from_folder(
        path_train, 
        valid_pct=0.2, 
        seed=42, 
        item_tfms=Resize(224), 
        batch_tfms=aug_transforms(), 
        valid_folder=path_valid
    )

    # Escolha a arquitetura desejada e crie o Learner
    arch = 'shufflenet_v2_x1_0'  # ou 'vgg16', 'mobilenet_v2', etc.
    learner = get_learner(arch, dls)

    if torch.backends.mps.is_available():
        print("Usando dispositivo MPS.")
        learner.to(torch.device('mps'))
    else:
        print("Dispositivo MPS não disponível, usando CPU.")
        learner.to(torch.device('cpu'))

    # Exemplo de uso
    epochs = 5
    lr = 1e-3

    # Callback para salvar o modelo e permitir continuar o treinamento
    checkpoint_callback = SaveModelCallback(monitor='accuracy', fname='best_model')
    train_with_progress(learner, epochs, lr, cbs=[checkpoint_callback])

    # Carregar o melhor modelo se necessário
    learner.load('best_model')

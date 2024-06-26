# Iov IDS

To run this project, you will need create a virtual env of python or run with docker.

### To run with virtual env

Please, use the cmd prompt, if you are using Windows.

run: `python -m venv env`

run (windows): `env\Scripts\activate.bat`
run (mac/linux): `source env/bin/activate`

run: `pip install --upgrade pip`
run: `pip install --no-cache-dir -r requirements.txt`

### Generating images

- Move the dataset files (attack_free.txt, dos.txt and fuzzy.txt) to `data/prod/datasets` folder.
- run: `python src/main.py gen_imgs`

### Trainning

To train ShuffleNetV2 run: `python src/main.py train shuffleNetV2`  
To train VGG16 run: `python src/main.py train vgg16`


If is running in MAC OS, please set `export PYTORCH_ENABLE_MPS_FALLBACK=1` on terminal to use CPU.

To active the cuda cores in NVIDIA, follow this tutorial https://youtu.be/r7Am-ZGMef8?si=65P0uvfMOMjf4Ehc or install:

- Visual Studio 2019: https://visualstudio.microsoft.com/vs...
- NVIDIA CUDA: https://developer.nvidia.com/cuda-too...
- NVIDIA CUDNN:  https://developer.nvidia.com/rdp/cudn...
- PyTorch : https://pytorch.org/get-started/locally/
from dotenv import load_dotenv
import os

load_dotenv()

def generate_images():
    print("generating images..")

def start_train():
    print("training...")

command = os.getenv('TYPE_RUN')

if command == 'SHOW_MESSAGE':
    print('It\'s Working!')

if command == 'GEN_IMAGES':
    generate_images()

if command == 'TRAIN':
    start_train()
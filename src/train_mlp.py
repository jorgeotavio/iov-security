import pandas as pd
import re
from fastai.tabular.all import *
from pathlib import Path

files = [
    (Path('data/prod/datasets/fuzzy.txt'), 0),
    (Path('data/prod/datasets/dos.txt'), 1),
    (Path('data/prod/datasets/attack_free.txt'), 2)
]

# Função para processar cada linha do arquivo
def process_line(line):
    match = re.match(r'Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)\s+\d+\s+DLC:\s+(\d+)\s+(.+)', line)
    if match:
        timestamp, msg_id, dlc, data = match.groups()
        data_split = data.split()
        return [float(timestamp), msg_id, int(dlc)] + [int(byte, 16) for byte in data_split]
    return None

# Lista para acumular todos os dados
all_data = []

# Processar cada arquivo e adicionar os dados na lista
for file_path, label in files:
    with open(file_path, 'r') as file:
        data = [process_line(line) for line in file.readlines() if process_line(line) is not None]
        for row in data:
            row.append(label)  # Adicionar o label à linha
        all_data.extend(data)  # Adicionar os dados processados à lista geral

# Criar o DataFrame
columns = ['Timestamp', 'ID', 'DLC'] + [f'Data_{i}' for i in range(8)] + ['target']  # Inclui 'target'
df = pd.DataFrame(all_data, columns=columns)

# Convertendo 'ID' para categórico
df['ID'] = df['ID'].astype('category')

# Configurar o DataLoader com Fastai
cont_names = ['Timestamp', 'DLC'] + [f'Data_{i}' for i in range(8)]  # Colunas contínuas
cat_names = ['ID']  # Colunas categóricas
dls = TabularDataLoaders.from_df(df, y_names='target', cat_names=cat_names, cont_names=cont_names)

# Definir e treinar o modelo MLP
learn = tabular_learner(dls, layers=[64, 32], metrics=accuracy)
learn.fit_one_cycle(10)

# Exemplo de previsão com novos dados
new_data = df.iloc[0]  # Exemplo de uma nova amostra
pred, pred_idx, probs = learn.predict(new_data)
print(f'Predição: {pred}, Índice: {pred_idx}, Probabilidades: {probs}')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Preprocess data function
def preprocess_data(files):
    data = []
    labels = []
    encoder = LabelEncoder()

    for file_path, label in files:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                try:
                    timestamp = float(parts[1])
                    data_id = parts[3]
                    dlc = int(parts[5])
                    data_hex = parts[7:]

                    # Convert hex data to decimal
                    data_dec = [int(byte, 16) for byte in data_hex]

                    # Pad data to ensure fixed length
                    if len(data_dec) < 8:
                        data_dec.extend([0] * (8 - len(data_dec)))

                    data.append(data_dec)
                    labels.append(label)
                except (ValueError, IndexError):
                    # Skip lines with invalid data
                    continue

    # Convert data to NumPy array
    data = np.array(data)

    # Reshape if necessary
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)  # Single feature per sample
    elif len(data.shape) == 2 and data.shape[1] == 1:
        data = data.reshape(-1)  # Flatten if only one feature

    # Encode labels
    labels = encoder.fit_transform(labels)

    data = StandardScaler().fit_transform(data)
    data = data.reshape(-1, 8, 1)  # Reshape for LSTM (samples, timesteps, features)
    return np.array(data), np.array(labels)

def get_dataloaders(data, labels, batch_size):
  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.shuffle(buffer_size=len(data))
  train_size = int(0.7 * len(dataset))
  train_dataset = dataset.take(train_size)
  val_dataset = dataset.skip(train_size)
  train_dataset = train_dataset.batch(batch_size)
  val_dataset = val_dataset.batch(batch_size)
  return train_dataset, val_dataset

def build_lstm_model(input_shape, num_classes):
  model = Sequential([
      LSTM(50, input_shape=input_shape, return_sequences=False),
      Dense(num_classes, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train_model(model, train_dataset, val_dataset, epochs, checkpoint_path):
  checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
  history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[checkpoint_callback])
  return history

def start():
  print('---- STARTING TRAIN ----')

  # Paths to training and validation data
  files = [
      (Path('data/prod/datasets/fuzzy.txt'), 0),
      (Path('data/prod/datasets/dos.txt'), 1),
      (Path('data/prod/datasets/attack_free.txt'), 2)
  ]

  # Preprocess the data
  data, labels = preprocess_data(files)

  # Split data into


if __name__ == '__main__':
  start()
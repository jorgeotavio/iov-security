import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from pathlib import Path
import os
import numpy as np

path = Path(os.getcwd())
data_dir = str(path) + "/data/"
train_dir = data_dir + 'train/'
val_dir = data_dir + 'validation/'

num_classes = 3
batch_size = 32
epochs = 10
use_gpu = True
train_data_percentage = 0.8

device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

base_model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1).to(device)
for param in base_model.parameters():
    param.requires_grad = False

class CustomShuffleNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomShuffleNet, self).__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.new_fc = nn.Linear(base_model.conv5[0].out_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.new_fc(x.view(x.size(0), -1))
        return x

model = CustomShuffleNet(base_model, num_classes).to(device)

optimizer = optim.Adam(model.new_fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_data(data_dir, transforms, percentage=100):
    dataset = datasets.ImageFolder(data_dir, transform=transforms)
    subset_size = int(len(dataset) * (percentage/100))
    indices = np.random.choice(range(len(dataset)), subset_size, replace=False)
    return Subset(dataset, indices)

train_data = load_data(train_dir, train_transforms, train_data_percentage)
val_data = load_data(val_dir, val_transforms, train_data_percentage)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4, prefetch_factor=2)

def train_and_validate(model, epochs, train_loader, val_loader, criterion, optimizer, device):
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}, Val Accuracy: {100 * correct / total}%')

train_and_validate(model, epochs, train_loader, val_loader, criterion, optimizer, device)

torch.save(model.state_dict(), data_dir+'result/result.pth')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import os
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

path = Path(os.getcwd())
data_dir = str(path) + "/data/"

train_dir = data_dir + 'images'
val_dir = data_dir + 'validation'

num_classes = 3
batch_size = 32
epochs = 10

base_model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
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

base_model = CustomShuffleNet(base_model, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.new_fc.parameters(), lr=0.001)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder(train_dir, transform=train_transforms)
val_data = ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

for epoch in range(epochs):
    base_model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer.zero_grad()
        outputs = base_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    base_model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = base_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}, Val Accuracy: {100 * correct / total}%')

torch.save(base_model.state_dict(), 'data/model-trained/result.pth')

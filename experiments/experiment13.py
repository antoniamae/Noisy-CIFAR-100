import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
import os
import timm
import numpy as np
import pandas as pd
from torch.backends import cudnn
from torch.cuda.amp import GradScaler
from torch import optim
from tqdm import tqdm

import wandb

wandb.init(project="cifar100-noisy", name="resnet50_e15")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
pin_memory = True
enable_half = torch.cuda.is_available()
scaler = GradScaler(enabled=enable_half)

class SimpleCachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = [x for x in dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class CIFAR100Noisy(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, noise_file=None):
        super(CIFAR100Noisy, self).__init__()
        self.dataset = CIFAR100(root=root, train=train, download=True)
        self.transform = transform
        self.noise_file = noise_file

        self.clean_labels = np.array(self.dataset.targets)

        if self.noise_file:
            noise_data = np.load(noise_file, allow_pickle=True)
            self.noisy_labels = noise_data.get('noisy_labels', self.clean_labels)
            self.noise_mask = noise_data.get('noise_mask', np.zeros_like(self.clean_labels, dtype=bool))
        else:
            self.noisy_labels = self.clean_labels.copy()
            self.noise_mask = np.zeros_like(self.clean_labels, dtype=bool)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        label = self.noisy_labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label, index


def get_transforms():
    train_transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    return train_transform, test_transform


def prepare_dataloader(root, batch_size, noise_file=None):
    train_transform, test_transform = get_transforms()

    train_dataset = CIFAR100Noisy(
        root=root,
        train=True,
        transform=train_transform,
        noise_file=noise_file
    )

    test_dataset = CIFAR100(
        root=root,
        train=False,
        transform=test_transform,
        download=False
    )

    train_dataset = SimpleCachedDataset(train_dataset)
    test_dataset = SimpleCachedDataset(test_dataset)

    return train_dataset, test_dataset


data_path = 'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100'
batch_size = 128
noise_file = "kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\CIFAR-100-noisy.npz"
train_dataset, test_dataset = prepare_dataloader(data_path, batch_size, noise_file)

train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )

test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

model = timm.create_model("resnet50", pretrained=True)
# model.fc = nn.Linear(512, 100)
model.fc = nn.Linear(2048, 100)
# model.fc = nn.Sequential(
#     nn.Dropout(0.3),
#     nn.Linear(model.fc.in_features, 100)
# )
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

def train():
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        running_loss += loss.item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(train_loader)
    return accuracy, avg_loss

@torch.no_grad()
def validate():
    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        running_loss += loss.item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(test_loader)
    return accuracy, avg_loss


# %%
@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


best = 0.0
num_epochs = 25

with tqdm(range(num_epochs)) as tbar:
    for epoch in tbar:
        train_acc, train_loss = train()
        val_acc, val_loss = validate()

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        tbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% Loss: {val_loss:.4f} | Best Val Acc: {best:.2f}%")

        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "best_val_accuracy": best
        })


# %%
data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("kaggle\\working\\submission9.csv", index=False)

wandb.save("kaggle\\working\\submission9.csv")
# %%
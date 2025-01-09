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

wandb.init(project="cifar100-noisy", name="resnet18_cifar10_e6")

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

class CIFAR100_noisy_fine(torch.utils.data.Dataset):
    def __init__(self, root, train, transform, download):
        cifar100 = CIFAR100(root=root, train=train, transform=transform, download=download)
        data, targets = zip(*cifar100)

        if train:
            # noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            noisy_label_file = "kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\CIFAR-100-noisy.npz"
            print(f"Calea completă către fișier: {noisy_label_file}")
            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomCrop((32, 32), padding=4),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True),
])

data_path = 'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100'
train_set = CIFAR100_noisy_fine(data_path, download=False, train=True, transform=basic_transforms)
test_set = CIFAR100_noisy_fine(data_path, download=False, train=False, transform=test_transforms)
train_set = SimpleCachedDataset(train_set)
test_set = SimpleCachedDataset(test_set)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

model = timm.create_model("hf_hub:grodino/resnet18_cifar10", pretrained=True)
model.fc = nn.Linear(512, 100)
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
num_epochs = 20

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
df.to_csv("kaggle\\working\\submission6.csv", index=False)

wandb.save("kaggle\\working\\submission6.csv")
# %%
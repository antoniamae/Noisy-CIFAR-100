# 22 - lr(m), wd(m), epochs changed
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
import os
import timm
import numpy as np
import pandas as pd
from torch.backends import cudnn
from torch.cuda.amp import GradScaler
from torch import optim
from tqdm import tqdm

import wandb

wandb.init(project="cifar100-noisy", name="experiment27")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
pin_memory = True
enable_half = torch.cuda.is_available()
scaler = GradScaler(enabled=enable_half)

class SimpleCachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = [(data, target) for data, target in dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CIFAR100_noisy_fine(torch.utils.data.Dataset):
    def __init__(self, root, train, transform, download):
        cifar100 = CIFAR100(root=root, train=train, transform=transform, download=download)
        data, targets = zip(*cifar100)

        if train:
            noisy_label_file = "kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\CIFAR-100-noisy.npz"
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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

data_path = 'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100'

train_dataset = CIFAR100_noisy_fine(root=data_path, train=True, transform=train_transform, download=False)
val_dataset = CIFAR100(root=data_path, train=False, transform=val_transform, download=False)

train_dataset = SimpleCachedDataset(train_dataset)
test_dataset = SimpleCachedDataset(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=pin_memory)

model = timm.create_model('resnet50', pretrained=True, num_classes=100, drop_rate=0.3)
model = model.to(device)

class AdaptiveGeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, initial_q=0.7, max_q=0.95, step=0.01):
        super().__init__()
        self.q = initial_q
        self.max_q = max_q
        self.step = step

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        loss = (1 - probs[range(probs.shape[0]), targets] ** self.q) / self.q
        self.q = min(self.q + self.step, self.max_q)  # Update q
        return loss.mean()

criterion = AdaptiveGeneralizedCrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

class MentorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

mentor_net = MentorNet(input_dim=100).to(device)
mentor_optimizer = optim.Adam(mentor_net.parameters(), lr=1e-2, weight_decay=5e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)


mentor_net.train()
for epoch in range(15):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        mentor_optimizer.zero_grad()
        mentor_outputs = mentor_net(model(inputs).detach())
        loss = nn.BCEWithLogitsLoss()(mentor_outputs, targets.float().unsqueeze(1))
        loss.backward()
        mentor_optimizer.step()


def train():
    model.train()
    mentor_net.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        mentor_inputs = model(inputs).detach()
        mentor_scores = mentor_net(mentor_inputs).squeeze()
        weights = torch.clamp(mentor_scores / (mentor_scores.sum() + 1e-6), min=0.01, max=1.0)

        mentor_loss = (weights * loss).sum()

        optimizer.zero_grad()
        mentor_optimizer.zero_grad()

        scaler.scale(mentor_loss).backward()
        scaler.step(optimizer)
        scaler.step(mentor_optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    accuracy = 100. * correct / total
    avg_loss = running_loss / total
    return accuracy, avg_loss



@torch.no_grad()
def validate():
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    accuracy = 100. * correct / total
    avg_loss = running_loss / total
    return accuracy, avg_loss

@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels

best = 0.0
num_epochs = 50

with tqdm(range(num_epochs)) as tbar:
    for epoch in tbar:
        train_acc, train_loss = train()
        val_acc, val_loss = validate()
        scheduler.step(val_acc)

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

data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("kaggle\\working\\submission27.csv", index=False)

wandb.save("kaggle\\working\\submission27.csv")

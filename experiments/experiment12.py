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
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.loss import LabelSmoothingCrossEntropy

wandb.init(project="cifar100-noisy", name="resnet50_improved_e14")

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


train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.2),
    v2.RandomRotation(15),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    v2.RandAugment(num_ops=2, magnitude=9),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), inplace=True)
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), inplace=True)
])

data_path = 'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100'
train_set = CIFAR100_noisy_fine(data_path, download=False, train=True, transform=train_transforms)
test_set = CIFAR100_noisy_fine(data_path, download=False, train=False, transform=test_transforms)
train_set = SimpleCachedDataset(train_set)
test_set = SimpleCachedDataset(test_set)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=256, pin_memory=pin_memory)


class ImprovedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, drop_rate=0.3)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        return self.backbone(x)


model = ImprovedModel().to(device)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train():
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        running_loss += loss.item()

    scheduler.step()
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


best = 0.0
patience = 10
epochs_without_improvement = 0
num_epochs = 100

with tqdm(range(num_epochs)) as tbar:
    for epoch in tbar:
        train_acc, train_loss = train()
        val_acc, val_loss = validate()

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        tbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.2f}% Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% Loss: {val_loss:.4f} | Best Val Acc: {best:.2f}%")

        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "best_val_accuracy": best
        })


@torch.inference_mode()
def inference():
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("kaggle/working/submission14.csv", index=False)
wandb.save("kaggle/working/submission14.csv")
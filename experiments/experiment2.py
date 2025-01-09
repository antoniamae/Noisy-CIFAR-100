# %%
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from typing import Optional, Callable
import os
import timm
import numpy as np
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
from sklearn.cluster import KMeans

# %%
import wandb

wandb.init(project="cifar100-noisy", name="resnet34_training")

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
pin_memory = True
enable_half = torch.cuda.is_available()  # Disable for CPU, it is slower!
scaler = GradScaler(enabled=enable_half)


# %%
def filter_noisy_labels(data, noisy_labels, threshold=0.8):
    print("hei1")
    features = np.array([np.array(image).flatten() for image in data])
    kmeans = KMeans(n_clusters=20, random_state=42).fit(features)
    confidences = kmeans.transform(features).min(axis=1)
    mask = confidences < np.percentile(confidences, threshold * 100)
    print("hei2")
    return [data[i] for i in range(len(data)) if mask[i]], [noisy_labels[i] for i in range(len(noisy_labels)) if
                                                            mask[i]]


# %%
class SimpleCachedDataset(Dataset):
    def __init__(self, dataset, runtime_transformers, cache):
        if cache:
            dataset = tuple([x for x in dataset])
        self.data = dataset
        self.runtime_transformers = runtime_transformers

    print("hei3")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i]
        if self.runtime_transformers is None:
            return image, label
        return self.runtime_transformers(image), label


# %%
class CIFAR100_noisy_fine(Dataset):
    def __init__(self, root: str, train: bool, transform: Optional[Callable], download: bool):
        cifar100 = CIFAR100(root=root, train=train, transform=transform, download=download)
        data, targets = tuple(zip(*cifar100))

        if train:
            # noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            noisy_label_file = "kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\CIFAR-100-noisy.npz"
            print(f"Calea completă către fișier: {noisy_label_file}")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(f"{type(self).__name__} need {noisy_label_file} to be used!")

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            data, targets = filter_noisy_labels(data, noise_file["noisy_label"])

        self.data = data
        self.targets = targets

    print("hei4")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]


# %%
runtime_transformers = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(size=32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])
basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])
print("hei5")
train_set = CIFAR100_noisy_fine(
    'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100', download=False,
    train=True, transform=basic_transforms)
test_set = CIFAR100_noisy_fine(
    'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100', download=False,
    train=False, transform=basic_transforms)
train_set = SimpleCachedDataset(train_set, runtime_transformers, False)
test_set = SimpleCachedDataset(test_set, None, False)
print("hei6")
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

# %%
print("hei7")
model = timm.create_model("resnet34", pretrained=True)
model.fc = nn.Linear(512, 100)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
print("hei8")


# %%
def train():
    print("hei9")
    model.train()
    correct = 0
    total = 0
    train_loss = 0.0
    print("hei10")
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("hei11")
    scheduler.step()
    return 100.0 * correct / total, train_loss / len(train_loader)


# %%
@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    print("hei12")
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("hei13")
    return 100.0 * correct / total, val_loss / len(test_loader)

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

# %%
best = 0.0
epochs = list(range(150))
print("hei14")
with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc, train_loss = train()
        val_acc, val_loss = val()
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "kaggle\\working\\best_model.pth")
        tbar.set_description(
            f"Epoch {epoch + 1}, Train: {train_acc:.2f}%, Loss: {train_loss:.4f}, Val: {val_acc:.2f}%, Loss: {val_loss:.4f}, Best: {best:.2f}%")
        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "best_val_accuracy": best
        })

print("hei15")
# %%
data = {
    "ID": [],
    "target": []
}

for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("kaggle\\working\\submission1.csv", index=False)

wandb.save("kaggle\\working\\submission1.csv")
# %%

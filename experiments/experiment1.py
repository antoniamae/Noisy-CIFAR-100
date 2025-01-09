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

# %%
import wandb

wandb.init(project="cifar100-noisy", name="resnet18_training")

# %%
device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True
scaler = GradScaler(device, enabled=enable_half)


# %%
class SimpleCachedDataset(Dataset):
    def __init__(self, dataset, runtime_transformers, cache):
        if cache:
            dataset = tuple([x for x in dataset])
        self.data = dataset
        self.runtime_transformers = runtime_transformers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i]
        if self.runtime_transformers is None:
            return image, label
        return self.runtime_transformers(image), label


# %%
class CIFAR100_noisy_fine(Dataset):
    """
    See https://github.com/UCSC-REAL/cifar-10-100n, https://www.noisylabels.com/ and `Learning with Noisy Labels
    Revisited: A Study Using Real-World Human Annotations`.
    """

    def __init__(
            self, root: str, train: bool, transform: Optional[Callable], download: bool
    ):
        cifar100 = CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
        data, targets = tuple(zip(*cifar100))

        if train:
            # noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            noisy_label_file = "kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\CIFAR-100-noisy.npz"
            print(f"Calea completă către fișier: {noisy_label_file}")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} need {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]


# %%
runtime_transformers = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(size=32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])
basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])
train_set = CIFAR100_noisy_fine(
    'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100', download=False,
    train=True, transform=basic_transforms)
test_set = CIFAR100_noisy_fine(
    'kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100\\fii-atnn-2024-project-noisy-cifar-100', download=False,
    train=False, transform=basic_transforms)
train_set = SimpleCachedDataset(train_set, runtime_transformers, False)
test_set = SimpleCachedDataset(test_set, None, False)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

# %%
model = timm.create_model("hf_hub:grodino/resnet18_cifar10", pretrained=True)
model.fc = nn.Linear(512, 100)
model = model.to(device)
# model = torch.jit.script(model)  # does not work for this model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, fused=True)


# %%
def train():
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


# %%
@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


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
with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train()
        val_acc = val()
        if val_acc > best:
            best = val_acc
        tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")

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
df.to_csv("kaggle\\working\\submission1.csv", index=False)

wandb.save("kaggle\\working\\submission1.csv")
# %%

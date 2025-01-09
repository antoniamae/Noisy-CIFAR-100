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

device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True  # Disable for CPU, it is slower!
scaler = GradScaler(device, enabled=enable_half)



class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        # Runtime transforms are not implemented in this simple cached dataset.
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]




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
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
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



basic_transforms = v2.Compose(
            [v2.Resize((224,224)),
             v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_transform = v2.Compose(
                [v2.Resize((224,224)),
                 v2.RandAugment(num_ops=1),
                 v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
train_set = CIFAR100_noisy_fine('D:\\kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100', download=False, train=True, transform=train_transform)
test_set = CIFAR100_noisy_fine('D:\\kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100', download=False, train=False, transform=basic_transforms)
train_set = SimpleCachedDataset(train_set)
test_set = SimpleCachedDataset(test_set)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)



import torch.optim.lr_scheduler as lr_scheduler

model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 100)
model = model.to(device)
# model = torch.jit.script(model)  # does not work for this model
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-5
        )


def train():
    model.train()
    correct = 0
    total = 0
    cutmix = v2.CutMix(num_classes=100, alpha=0.5)
    mixup = v2.MixUp(num_classes=100, alpha=0.5)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    # cutmix_transform = v2.CutMix(num_classes=100, alpha=1.0)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        inputs, targets = cutmix_or_mixup(inputs, targets)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets.argmax(dim=1)).sum().item()

    return 100.0 * correct / total


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


#
# best = 0.0
# epochs = list(range(16))
# model_save_path = "best_model.pth"
# warmup = 5
# learning_rate = 0.0001
# with tqdm(epochs) as tbar:
#     for epoch in tbar:
#         if epoch < warmup:
#             lr_scale = min(1., float(epoch + 1) / warmup)
#             for pg in optimizer.param_groups:
#                 pg['lr'] = learning_rate * lr_scale
#
#         train_acc = train()
#         val_acc = val()
#         if val_acc > best:
#             best = val_acc
#             torch.save(model.state_dict(), model_save_path)
#
#         scheduler.step()
#         tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")

model_save_path = "eff2_best_model.pth"
model.load_state_dict(torch.load(model_save_path))
data = {
    "ID": [],
    "target": []
}


for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
df.to_csv("submission.csv", index=False)
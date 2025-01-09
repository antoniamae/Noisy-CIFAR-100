import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler

import wandb
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from tqdm import tqdm

from preactresnet18 import PreActResNet18

pin_memory = True

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

class ELRLoss(nn.Module):
    def __init__(self, num_classes, num_samples, beta=0.7, lambda_=3.0):
        super().__init__()
        self.num_classes = num_classes
        self.NUM_SAMPLES = num_samples
        self.beta = beta
        self.lambda_ = lambda_
        self.history = torch.zeros(self.NUM_SAMPLES, self.num_classes).float().cuda()

    def forward(self, index, output, target):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)


        ce_loss = torch.mean(torch.sum(-target * F.log_softmax(output, dim=1), dim=1))


        reg = ((1 - (self.history[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lambda_ * reg

        return final_loss

    def update_hist(self, epoch, out_prob, index, mix_index=None, mixup_l=None):
        with torch.no_grad():
            if mix_index is None:
                self.history[index] = self.beta * self.history[index] + (1 - self.beta) * out_prob
            else:
                self.history[index] = self.beta * self.history[index] + (1 - self.beta) * (
                    mixup_l * out_prob + (1 - mixup_l) * out_prob[mix_index]
                )

class ELRPlus:
    def __init__(
            self,
            model1: nn.Module,
            model2: nn.Module,
            num_classes: int = 100,
            num_samples: int = 50000,
            beta: float = 0.9,
            lambda_: float = 7.0,
            mixup_alpha: float = 1.0,
            ema_alpha: float = 0.997,
            ema_update: bool = True,
            ema_step: int = 40000,
            num_epochs: int = 250,
            num_warmup_epochs: int = 0,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.model_ema1 = deepcopy(model1).to(device)
        self.model_ema2 = deepcopy(model2).to(device)

        for param in self.model_ema1.parameters():
            param.requires_grad = False
        for param in self.model_ema2.parameters():
            param.requires_grad = False

        self.device = device
        print(device)
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.ema_alpha = ema_alpha
        self.ema_update = ema_update
        self.ema_step = ema_step
        self.num_epochs = num_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.global_step = 0

        self.criterion1 = ELRLoss(num_classes, num_samples, beta, lambda_).to(device)
        self.criterion2 = ELRLoss(num_classes, num_samples, beta, lambda_).to(device)

    def mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1 - lam)
            batch_size = x.size()[0]
            mix_index = torch.randperm(batch_size).to(self.device)

            mixed_x = lam * x + (1 - lam) * x[mix_index, :]
            mixed_target = lam * y + (1 - lam) * y[mix_index, :]
            return mixed_x, mixed_target, lam, mix_index
        else:
            return x, y, 1, None

    def update_ema_variables(self, model, model_ema, global_step, alpha=0.997):
        if alpha == 0:
            for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data = param.data
        else:
            if self.ema_update:
                alpha = sigmoid_rampup(global_step + 1, self.ema_step) * alpha
            else:
                alpha = min(1 - 1 / (global_step + 1), alpha)
            for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def train_epoch(self, epoch, train_loader, optimizer1, optimizer2):
        self.model1.train()
        self.model2.train()

        epoch_loss = 0
        correct = 0
        total = 0

        start_time = time.time()
        with tqdm(train_loader, desc=f'Train epoch {epoch}') as progress:
            for batch_idx, (data, target, index) in enumerate(progress):
                data, target = data.to(self.device), target.to(self.device)

                target_one_hot = torch.zeros(len(target), self.num_classes).to(self.device)
                target_one_hot.scatter_(1, target.view(-1, 1), 1)

                if np.random.rand() < 0.5:
                    data, target = self.apply_cutmix(data, target_one_hot)

                optimizer1.zero_grad()
                output1 = self.model1(data)
                loss1 = F.cross_entropy(output1, target)
                loss1.backward()
                optimizer1.step()

                optimizer2.zero_grad()
                output2 = self.model2(data)
                loss2 = F.cross_entropy(output2, target)
                loss2.backward()
                optimizer2.step()

                epoch_loss += (loss1.item() + loss2.item()) / 2
                preds = (output1 + output2).argmax(dim=1)
                if target.ndim > 1:
                    correct += preds.eq(target.argmax(dim=1)).sum().item()
                else:
                    correct += preds.eq(target).sum().item()

                total += target.size(0)
                total += target.size(0)

        end_time = time.time()
        train_acc = 100. * correct / total

        wandb.log({
            "train_loss": epoch_loss / len(train_loader),
            "train_accuracy": train_acc,
            "epoch": epoch,
            "time_per_epoch": end_time - start_time
        })

    def apply_cutmix(self, data, target, alpha=1.0):
        """Applies CutMix augmentation."""
        lam = np.random.beta(alpha, alpha)
        batch_size = data.size(0)
        indices = torch.randperm(batch_size).to(data.device)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]


        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
        target = lam * target + (1 - lam) * target[indices]
        return data, target

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)


        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def evaluate(self, test_loader, epoch=None):
        self.model1.eval()
        self.model2.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target, index in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output1 = self.model1(data)
                output2 = self.model2(data)
                output = 0.5 * (output1 + output2)

                loss = F.cross_entropy(output, target)
                total_loss += loss.item()

                preds = (output1 + output2).argmax(dim=1)

                if target.ndim > 1:
                    correct += preds.eq(target.argmax(dim=1)).sum().item()
                else:
                    correct += preds.eq(target).sum().item()

                total += target.size(0)

        val_acc = 100. * correct / total
        val_loss = total_loss / len(test_loader)


        if epoch is not None:
            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch
            })

        return val_acc

    import os

    def train(self, train_loader, test_loader, optimizer1, optimizer2, scheduler1, scheduler2,
              save_path="best_model_elrrr.pth"):
        best_acc = 0

        print("Starting main training...")
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch, train_loader, optimizer1, optimizer2)
            test_acc = self.evaluate(test_loader)
            print(f'Epoch {epoch}, Test Accuracy: {test_acc:.2f}%')



            wandb.log({
                "test_accuracy": test_acc,
                "epoch": epoch
            })


            if test_acc > best_acc:
                best_acc = test_acc
                best_state = {
                    "model1_state_dict": self.model1.state_dict(),
                    "model2_state_dict": self.model2.state_dict(),
                    "model_ema1_state_dict": self.model_ema1.state_dict(),
                    "model_ema2_state_dict": self.model_ema2.state_dict(),
                    "optimizer1_state_dict": optimizer1.state_dict(),
                    "optimizer2_state_dict": optimizer2.state_dict(),
                    "scheduler1_state_dict": scheduler1.state_dict(),
                    "scheduler2_state_dict": scheduler2.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc
                }
                torch.save(best_state, save_path)
                print(f"Best model saved with accuracy: {best_acc:.2f}%")
                scheduler1.step(test_acc)
                scheduler2.step(test_acc)


        checkpoint = torch.load(save_path)
        self.model1.load_state_dict(checkpoint["model1_state_dict"])
        self.model2.load_state_dict(checkpoint["model2_state_dict"])
        self.model_ema1.load_state_dict(checkpoint["model_ema1_state_dict"])
        self.model_ema2.load_state_dict(checkpoint["model_ema2_state_dict"])

        return best_acc


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, target = self.data[i]
        return data, target, i



class CIFAR100_noisy_fine(Dataset):
    def __init__(self, root: str, train: bool, transform: Optional[Callable], download: bool):
        cifar100 = CIFAR100(root=root, train=train, transform=transform, download=download)
        data, targets = tuple(zip(*cifar100))

        if train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(f"{type(self).__name__} need {noisy_label_file} to be used!")

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i], i

basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True)
])
train_transform = v2.Compose([
    v2.RandAugment(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])
train_set = CIFAR100_noisy_fine(
    'D:\\kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100', download=False,
    train=True, transform=train_transform)
test_set = CIFAR100_noisy_fine(
    'D:\\kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100', download=False,
    train=False, transform=basic_transforms)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

wandb.init(
    project="elr_plus_real_cifar100",
    config={
        "learning_rate": 0.001,
        "batch_size": 128,
        "num_epochs": 100,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "weight_decay": 5e-4,
        "mixup_alpha": 1.0,
        "upsample": 128
    }
)

from timm import create_model

# model1 = PreActResNet18(num_classes=100).to("cuda")
# model2 = PreActResNet18(num_classes=100).to("cuda")

model1 = create_model("resnet18", pretrained=True, num_classes= 100).to("cuda")
model1 = nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                model1
            )

model2 = create_model("resnet18", pretrained=True, num_classes= 100).to("cuda")
model2 = nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                model2
            )


optimizer1 = optim.AdamW(model1.parameters(), lr=0.001, weight_decay=5e-4)
optimizer2 = optim.AdamW(model2.parameters(), lr=0.001, weight_decay=5e-4)


# scheduler1 = MultiStepLR(optimizer1, milestones=[200], gamma=0.1)
# scheduler2 = MultiStepLR(optimizer2, milestones=[200], gamma=0.1)

scheduler1 = lr_scheduler.ReduceLROnPlateau(
    optimizer1, mode='max', factor=0.5, patience=5, threshold=1e-4, cooldown=2, min_lr=1e-6
)
scheduler2 = lr_scheduler.ReduceLROnPlateau(
    optimizer2, mode='max', factor=0.5, patience=5, threshold=1e-4, cooldown=2, min_lr=1e-6
)


trainer = ELRPlus(
    model1=model1,
    model2=model2,
    num_classes=100,
    num_samples=len(train_set)
)
start_training_time = time.time()
best_accuracy = trainer.train(train_loader, test_loader, optimizer1, optimizer2, scheduler1, scheduler2)
end_training_time = time.time()

wandb.log({"total_training_time": end_training_time - start_training_time})


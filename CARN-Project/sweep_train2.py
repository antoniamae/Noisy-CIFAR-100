import json
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from torch import nn, Tensor
import torch.nn.functional as F
import random

import torch.backends.cudnn

from typing import Literal, cast, Optional, Callable, Iterable

import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torchvision.datasets import CIFAR100
from torchvision.transforms.v2 import CutMix, MixUp

from torchvision.transforms import v2
from timm import create_model
import os
import pickle
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

import torch.optim.lr_scheduler as lr_scheduler

pin_memory = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def cache_dataset(dataset_class, data_dir, cache_dir='./cache', train=True):
    os.makedirs(cache_dir, exist_ok=True)
    subset = 'train' if train else 'test'
    cache_path = os.path.join(cache_dir, f'{dataset_class.__name__}_{subset}.pkl')

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = dataset_class(root=data_dir, train=train, download=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    return data


def get_data_augmentation(scheme="basic", dataset="CIFAR"):
    if dataset == "CIFAR":
        if scheme == "basic":
            train_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                          v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
        elif scheme == "random_flip":
            train_transform = v2.Compose(
                [v2.RandomHorizontalFlip(), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
        elif scheme == "random_crop_flip":
            train_transform = v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(), v2.ToImage(),
                                          v2.ToDtype(torch.float32, scale=True),
                                          v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
        elif scheme == "randaugment":
            train_transform = v2.Compose(
                [v2.Resize((224, 224)), v2.RandAugment(num_ops=1), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        elif scheme == "autoaugment":
            train_transform = v2.Compose(
                [v2.Resize((224, 224)), AutoAugment(policy=AutoAugmentPolicy.CIFAR10), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        elif scheme == "combined":
            train_transform = v2.Compose(
                [v2.RandomResizedCrop(32, scale=(0.8, 1.0)), v2.RandomHorizontalFlip(), v2.RandomRotation(15),
                 v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), v2.RandAugment(), v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
        elif scheme == "combined2":
            train_transform = v2.Compose(
                [AutoAugment(policy=AutoAugmentPolicy.CIFAR10), v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(),
                 v2.ColorJitter(brightness=0.2, contrast=0.2),
                 v2.RandomRotation(15), v2.AutoAugment(), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.5,), (0.5,))])
        elif scheme == "combined_resize":
            train_transform = v2.Compose([
                v2.Resize((64, 64)),
                v2.RandomResizedCrop(64, scale=(0.8, 1.0)),
                v2.RandomRotation(15),
                v2.RandomHorizontalFlip(),
                v2.RandAugment(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif scheme == "combined_resize2":
            train_transform = v2.Compose([
                v2.RandomRotation(10),
                v2.RandomResizedCrop(32, scale=(0.9, 1.1)),
                v2.RandomHorizontalFlip(),
                v2.RandomAffine(degrees=0, shear=10),
                v2.RandomCrop(32, padding=3),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            raise ValueError(f"Augmentation scheme '{scheme}' not supported for CIFAR.")
        test_transform = v2.Compose(
            [v2.Resize((224, 224)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

    else:
        raise ValueError(f"Dataset '{dataset}' not supported.")

    return train_transform, test_transform


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
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


def load_data(batch_size=64, scheme="basic", custom_transforms=None):
    if custom_transforms:
        train_transform, test_transform = custom_transforms
    else:
        train_transform, test_transform = get_data_augmentation(scheme=scheme, dataset="CIFAR")

    try:
        train_set = CIFAR100_noisy_fine(
            'D:\\kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100', download=False,
            train=True, transform=train_transform)
        test_set = CIFAR100_noisy_fine(
            'D:\\kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100', download=False,
            train=False, transform=test_transform)
        # pseudo_set = CIFAR100_noisy_fine(
        #     'D:\\kaggle\\input\\fii-atnn-2024-project-noisy-cifar-100', download=False,
        #     train=True, transform=test_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    train_set.transform = train_transform
    test_set.transform = test_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)
    # pseudo_loader = DataLoader(
    #     pseudo_set,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     pin_memory = False
    # )

    return train_loader, test_loader


def get_model(dataset, model_name, num_classes, input_size=None, hidden_layers=None, pretrained=True):
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        if model_name in ['resnet18', 'resnet18_resize']:
            model = create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
        elif model_name in ['resnet50', 'resnet50_resize', 'resnet50_try']:
            model = create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
        elif model_name in ['resnet101', 'resnet101_resize']:
            model = create_model("resnet101", pretrained=pretrained, num_classes=num_classes)
        elif model_name == 'resnet18_cifar10':
            model = create_model("hf_hub:edadaltocg/resnet18_cifar10", pretrained=False, num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        elif model_name == 'byol':
            model = create_model("resnet50", pretrained=False, num_classes=num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            model_path = '/kaggle/input/byol-resnet50/pytorch/default/1/byol_pretrained_resnet50.pth'
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("BYOL ResNet50 model loaded successfully.")
        elif model_name == 'byol2':
            model = create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
            model = nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                model
            )
            model_path = '/kaggle/input/byol-resnet50-resize/pytorch/default/1/byol_pretrained_resnet50_resize.pth'
            state_dict = torch.load(model_path, map_location='cuda')
            model.load_state_dict(state_dict)
        elif model_name == "eff":
            model = create_model('efficientnet_b3', pretrained=pretrained, num_classes=100)
            # upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
            model = nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                model
            )
        elif model_name == 'eff_ns':
            model = create_model("tf_efficientnet_b0_ns", pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, 100)
        else:
            raise ValueError(
                f"Model '{model_name}' is not supported for CIFAR.")
        if pretrained and model_name in ['resnet18_resize', 'resnet50_resize']:
            model = nn.Sequential(
                nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
                model
            )
        if pretrained and model_name in ['resnet18_try', 'resnet50_try']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()


    else:
        raise ValueError(f"Dataset '{dataset}' is not supported. Choose 'CIFAR10', 'CIFAR100', or 'MNIST'.")

    return model


import torch




import torch.optim as optim


def get_optimizer(optimizer_name, model_parameters, lr=0.001, momentum=0.9, weight_decay=0.0, nesterov=True):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=lr)
    elif optimizer_name == 'sgd_momentum':
        return optim.SGD(model_parameters, lr=lr, momentum=momentum)
    elif optimizer_name == 'sgd_nesterov':
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, nesterov=nesterov)
    elif optimizer_name == 'sgd_weight_decay':
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    # elif optimizer_name == "sam_adamw":
    #     base_opt= optim.AdamW
    #     return SAM( model_parameters,base_opt, lr=lr, rho=0.05, weight_decay=weight_decay)
    # rho=0.05,

    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")


def get_scheduler(optimizer, scheduler_name, **kwargs):
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'steplr':
        return lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.1))

    elif scheduler_name == 'reducelronplateau':
        mode_str = kwargs.get('mode', 'min')
        if mode_str not in ['min', 'max']:
            raise ValueError("Invalid mode for ReduceLROnPlateau: must be 'min' or 'max'")
        mode = cast(Literal["min", "max"], mode_str)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10)
        )

    elif scheduler_name == 'cosineannealinglr':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 50),
            eta_min=kwargs.get('eta_min', 0)
        )

    elif scheduler_name == 'cosineannealingwarmrestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('t_0', 10),
            T_mult=kwargs.get('t_mult', 1),
            eta_min=kwargs.get('eta_min', 0)
        )

    elif scheduler_name == 'exponentiallr':
        return lr_scheduler.ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.9))

    elif scheduler_name == 'linearlr':
        return lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 0.0),
            total_iters=kwargs.get('total_iters', 100)
        )

    elif scheduler_name == 'none':
        return None

    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not supported.")


def early_stopping(current_score, best_score, patience_counter, patience_early_stopping, min_delta=0.0, mode="max"):
    print("heeei", best_score, current_score, mode, patience_counter)
    if best_score is None:
        best_score = current_score
        return False, best_score, patience_counter

    if mode == "min":
        improvement = best_score - current_score > min_delta
    elif mode == "max":
        improvement = current_score - best_score > min_delta
    else:
        raise ValueError("Mode should be 'min' or 'max'")

    if improvement:
        best_score = current_score
        patience_counter = 0
    else:
        patience_counter += 1

    early_stop = patience_counter >= patience_early_stopping
    return early_stop, best_score, patience_counter


@torch.inference_mode()
def validate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(test_loader.dataset)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy


from torch.amp import autocast, GradScaler


def generate_pseudo_labels(model, loader, device, confidence_threshold=0.9):
    pseudo_data = []
    pseudo_labels = []
    model.eval()
    with torch.inference_mode():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, preds = probabilities.max(dim=1)

            confident_indices = max_probs > confidence_threshold
            pseudo_data.append(inputs[confident_indices])
            pseudo_labels.append(preds[confident_indices])

    if pseudo_data and pseudo_labels:
        return torch.cat(pseudo_data), torch.cat(pseudo_labels)
    return None, None

from torch.utils.data import Dataset
device = "cuda"
import torch

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        device = "cuda"

        if idx < len(self.dataset1):
            data, label = self.dataset1[idx]
        else:
            data, label = self.dataset2[idx - len(self.dataset1)]

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return data.to(device), label.to(device)


def train_model(model, train_loader, test_loader, device, num_epochs, optimizer, num_classes, scheduler_mode=None,
                scheduler=None, patience_early_stopping=5, min_delta=0.0, early_stop_mode="max", learning_rate=0.1,
                warmup=0, grad_alpha=1.0, use_cutmix=True, use_mixup=True, alpha=1.0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = GeneralizedCrossEntropyLoss(q=0.7)

    scaler = GradScaler(device)

    best_val_score = None
    best_train_score = None
    patience_counter = 0
    best_val_accuracy = 0.0
    best_train_accuracy = 0.0
    best_val_loss = 100.0
    best_train_loss = 100.0
    wandb.watch(model, log="all", log_freq=10)
    alpha = float(alpha)
    cutmix = v2.CutMix(num_classes=num_classes, alpha=alpha)
    mixup = v2.MixUp(num_classes=num_classes, alpha=alpha)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    rand = random.randint(1000, 9999)
    file_path = f"eff_best_model.pth"
    print("saving", file_path)
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch ", epoch)
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        if epoch < warmup:
            lr_scale = min(1., float(epoch + 1) / warmup)
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate * lr_scale

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


            if use_cutmix and use_mixup:
                inputs, labels = cutmix_or_mixup(inputs, labels)
            elif use_cutmix:
                inputs, labels = cutmix(inputs, labels)
            elif use_mixup:
                inputs, labels = mixup(inputs, labels)



            optimizer.zero_grad()
            with autocast(device):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()



            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            if use_cutmix or use_mixup:
                correct += predicted.eq(labels.argmax(dim=1)).sum().item()
            else:
                correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = validate_model(model, test_loader, criterion, device)
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), file_path)
            print(f"Best model saved with accuracy: {best_val_accuracy:.2f}%")
        print(f"Epoch {epoch + 1}/{num_epochs} - ")
        print(f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - ")
        print(f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'time_per_epoch':time.time() - start_time
        })

        val_score = val_loss if early_stop_mode == "min" else val_accuracy
        train_score = train_loss if early_stop_mode == "min" else train_accuracy


        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            if scheduler_mode == "max":
                scheduler.step(val_accuracy)
            elif scheduler_mode == "min":
                scheduler.step(val_loss)
        elif scheduler:
            scheduler.step()

        early_stop, best_train_score, patience_counter = early_stopping(
            current_score=train_score,
            best_score=best_train_score,
            patience_counter=patience_counter,
            patience_early_stopping=patience_early_stopping,
            min_delta=min_delta,
            mode=early_stop_mode
        )

        if early_stop:
            print("Early stopping triggered. Stopping training.")
            break



    print("Training complete.")

# def train_model(model, train_loader, test_loader, pseudo_loader, device, num_epochs, optimizer, num_classes,
#                 scheduler_mode=None, scheduler=None, patience_early_stopping=5, min_delta=0.0, early_stop_mode="max",
#                 learning_rate=0.1, warmup=0, grad_alpha=1.0, use_cutmix=True, use_mixup=True, alpha=1.0,
#                 confidence_threshold=0.9):
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#     scaler = GradScaler(device)
#
#     best_val_accuracy = 0.0
#     best_train_accuracy = 0.0
#     best_val_loss = float("inf")
#     patience_counter = 0
#
#     cutmix = v2.CutMix(num_classes=num_classes, alpha=alpha)
#     mixup = v2.MixUp(num_classes=num_classes, alpha=alpha)
#     cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
#
#     pseudo_dataset = None
#
#     file_path = "eff_best_model.pth"
#     print("Saving model to", file_path)
#
#     for epoch in range(num_epochs):
#         start_time = time.time()
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#
#         model.train()
#         running_loss = 0.0
#         correct, total = 0, 0
#
#         if epoch < warmup:
#             lr_scale = min(1.0, float(epoch + 1) / warmup)
#             for pg in optimizer.param_groups:
#                 pg["lr"] = learning_rate * lr_scale
#
#
#         if pseudo_dataset:
#             combined_dataset = CombinedDataset(train_loader.dataset, pseudo_dataset)
#         else:
#             combined_dataset = train_loader.dataset
#
#
#         # for inputs, labels in combined_dataset:
#         #     inputs, labels = inputs.to(device), labels.to(device)
#
#         combined_loader = DataLoader(
#             combined_dataset,
#             batch_size=train_loader.batch_size,
#             shuffle=True,
#             num_workers=train_loader.num_workers,
#             pin_memory = False
#         )
#
#
#
#         for inputs, labels in combined_loader:
#             torch.cuda.empty_cache()
#
#             inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#
#
#             if use_cutmix and use_mixup:
#                 inputs, labels = cutmix_or_mixup(inputs, labels)
#             elif use_cutmix:
#                 inputs, labels = cutmix(inputs, labels)
#             elif use_mixup:
#                 inputs, labels = mixup(inputs, labels)
#
#
#
#             optimizer.zero_grad()
#             with autocast(device):
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#
#             if use_cutmix or use_mixup:
#                 correct += predicted.eq(labels.argmax(dim=1)).sum().item()
#             else:
#                 correct += predicted.eq(labels).sum().item()
#
#         train_loss = running_loss / len(combined_loader.dataset)
#         train_accuracy = 100 * correct / total
#
#         val_loss, val_accuracy = validate_model(model, test_loader, criterion, device)
#
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             torch.save(model.state_dict(), file_path)
#             print(f"Best model saved with validation accuracy: {best_val_accuracy:.2f}%")
#
#         print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
#         print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
#
#         wandb.log({
#             "epoch": epoch + 1,
#             "train_loss": train_loss,
#             "train_accuracy": train_accuracy,
#             "val_loss": val_loss,
#             "val_accuracy": val_accuracy,
#             "time_per_epoch": time.time() - start_time
#         })
#
#         if early_stop_mode == "max" and val_accuracy <= best_val_accuracy - min_delta:
#             patience_counter += 1
#         elif early_stop_mode == "min" and val_loss >= best_val_loss + min_delta:
#             patience_counter += 1
#         else:
#             patience_counter = 0
#
#         if patience_counter >= patience_early_stopping:
#             print("Early stopping triggered. Stopping training.")
#             break
#
#         if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(val_accuracy if scheduler_mode == "max" else val_loss)
#         elif scheduler:
#             scheduler.step()
#
#         with torch.no_grad():
#             pseudo_inputs, pseudo_labels = generate_pseudo_labels(model, pseudo_loader, device, confidence_threshold)
#         if pseudo_inputs is not None:
#             pseudo_inputs = pseudo_inputs.to(device)
#             pseudo_labels = pseudo_labels.to(device)
#             pseudo_dataset = TensorDataset(pseudo_inputs, pseudo_labels)
#
#
#     print("Training complete.")


@torch.inference_mode()
def ensemble_validate(model_paths, test_loader, criterion, device):

    models = []
    for path in model_paths:
        model = create_model("tf_efficientnet_b0_ns", pretrained=False, num_classes=100)  # Adjust as needed
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        models.append(model)

    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)


        outputs = torch.zeros((inputs.size(0), 100), device=device)
        for model in models:
            outputs += F.softmax(model(inputs), dim=1)
        outputs /= len(models)

        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(test_loader.dataset)
    val_accuracy = 100 * correct / total
    print(f"Ensemble Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    return val_loss, val_accuracy


import wandb

@torch.inference_mode()
def inference(model, test_loader):
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device, enabled=True):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


def sweep_train():
    wandb.init()
    config = wandb.config
    run_name = f"aaa_alpha_0.5_diff_normal_5_warmup_optimizer-{config.optimizer_config['optimizer']}_lr-{config.optimizer_config['learning_rate']}_scheduler-{config.scheduler}_aug-{config.augmentation_scheme}_numops_1_model-{config.model_name}"
    wandb.run.name = run_name
    wandb.run.save()
    try:
        train_loader, test_loader = load_data(
            batch_size=config.batch_size,
            scheme=config.augmentation_scheme,
        )
        model = get_model(
            dataset=config.dataset,
            model_name=config.model_name,
            num_classes=config.num_classes
        )
        optimizer_name = config.optimizer_config["optimizer"]
        learning_rate = config.optimizer_config["learning_rate"]

        optimizer = get_optimizer(
            optimizer_name=optimizer_name,
            model_parameters=model.parameters(),
            lr=learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov
        )
        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_name=config.scheduler,
            t_max=config.get('t_max', 200),
            eta_min=config.get('eta_min', 0),
            step_size=config.get('step_size', 10),
            gamma=config.get('gamma', 0.1),
            patience=config.get('scheduler_patience', 10),
            factor=config.get('factor', 0.1)
        )

        # Uncomment the following line to enable training
        # train_model(
        #     model=model,
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     device=get_device(),
        #     num_epochs=config.num_epochs,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     patience_early_stopping=config.patience_early_stopping,
        #     min_delta=config.min_delta,
        #     early_stop_mode=config.stop_mode,
        #     learning_rate=learning_rate,
        #     num_classes=config.num_classes,
        #     use_cutmix=config.use_cutmix,
        #     use_mixup=config.use_mixup,
        #     alpha=config.alpha,
        #     warmup=config.warmup,
        # )

        model_paths = ["eff_best_model.pth", "eff2_best_model.pth"]
        models = []

        base_model = create_model("tf_efficientnet_b0_ns", pretrained=False, num_classes=100)
        val_loss, val_accuracy = ensemble_validate(
            model_paths=model_paths,
            test_loader=test_loader,
            criterion=nn.CrossEntropyLoss(),
            device="cuda"
        )
        print("Val accuracy:", val_accuracy)
        print("Test accuracy:", val_loss)
        labels = ensemble_inference(model_paths, base_model, test_loader, "cuda")

        data = {
            "ID": list(range(len(labels))),
            "target": labels,
        }
        df = pd.DataFrame(data)
        df.to_csv("submission.csv", index=False)
        print("Submission saved to 'submission.csv'.")

    finally:
        wandb.finish()


@torch.inference_mode()
def ensemble_inference(model_paths, base_model, test_loader, device):
    """
    Perform inference using an ensemble of models loaded from different .pth files.

    Args:
        model_paths: List of paths to the .pth files for the ensemble models.
        base_model: A base model architecture to load the weights into.
        test_loader: DataLoader for the test dataset.
        device: Device to run the models on.

    Returns:
        List of predicted labels.
    """
    # Load all models into memory
    models = []
    for path in model_paths:
        model = deepcopy(base_model)  # Start from the same base architecture
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)

        # Combine predictions from all models
        outputs = torch.zeros((inputs.size(0), models[0].num_classes), device=device)
        for model in models:
            with torch.autocast(device):
                outputs += F.softmax(model(inputs), dim=1)  # Accumulate probabilities
        outputs /= len(models)  # Average the predictions

        predicted = outputs.argmax(dim=1).tolist()
        labels.extend(predicted)

    return labels


def load_config(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".json":
        with open(file_path, 'r') as f:
            config = json.load(f)
    elif ext in {".yaml", ".yml"}:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON or YAML.")
    return config



config_file_path = "sweep_config2.json"
sweep_config = load_config(config_file_path)

sweep_id = wandb.sweep(sweep_config, project="training-cifar100-noisy-oooooooooooooooooo")

wandb.agent(sweep_id, sweep_train)





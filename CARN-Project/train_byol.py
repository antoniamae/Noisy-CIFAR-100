
import json

import numpy as np
import yaml
from torch import nn, Tensor
import torch.nn.functional as F
import random
import torch
from byol_pytorch import BYOL

import torch.backends.cudnn


from typing import Literal, cast, Optional, Callable

import torch
import torchvision
from torch import nn
from torch.export import export
from torchvision.datasets import CIFAR100
from torchvision.transforms.v2 import CutMix, MixUp

from torchvision.transforms import v2
from timm import create_model
import os
import pickle
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

import torch.optim.lr_scheduler as lr_scheduler
pin_memory = True

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
                [v2.RandAugment(), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
        elif scheme == "autoaugment":
            train_transform = v2.Compose(
                [AutoAugment(policy=AutoAugmentPolicy.CIFAR10), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

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
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    else:
        raise ValueError(f"Dataset '{dataset}' not supported.")

    return train_transform, test_transform

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

def load_data( batch_size=64, scheme="basic", custom_transforms=None):
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
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    train_set.transform = train_transform
    test_set.transform = test_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

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

        else:
            raise ValueError(
                f"Model '{model_name}' is not supported for CIFAR.")
        if pretrained and model_name in ['resnet18_resize', 'resnet50_resize']:
            model = nn.Sequential(
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False),
                model
            )
        if pretrained and model_name in ['resnet18_try', 'resnet50_try']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()


    else:
        raise ValueError(f"Dataset '{dataset}' is not supported. Choose 'CIFAR10', 'CIFAR100', or 'MNIST'.")

    return model


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
        # base_opt= optim.AdamW
        # return SAM( model_parameters,base_opt, lr=lr, rho=0.05, weight_decay=weight_decay)
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




def train_model(model, train_loader, test_loader, device, num_epochs):
    model = model.to(device)


    learner = BYOL(
        model[1],
        image_size=128,
        hidden_layer='global_pool',
        use_momentum=False
    ).to(device)

    optimizer = torch.optim.SGD(learner.parameters(), lr=0.01)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0


        for batch in train_loader:
            images, _ = batch
            images = images.to(device)

            with autocast(device_type=device):
                loss = learner(images)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(learner.state_dict(), f'byol_epoch_{epoch + 1}_resize.pth')

    print("Pretraining complete. Saving model.")
    torch.save(model.state_dict(), 'byol_pretrained_resnet50_resize.pth')






import wandb


def sweep_train(config):
    # wandb.init()
    # config = wandb.config
    try:
        train_loader, test_loader = load_data(
            batch_size=64,
            scheme="combined",
        )
        model = get_model(
            dataset="CIFAR100",
            model_name="resnet18_resize",
            num_classes=100
        )
        # optimizer_name = config.optimizer_config["optimizer"]
        # learning_rate = config.optimizer_config["learning_rate"]

        print("kkaka")
        # optimizer = get_optimizer(
        #     optimizer_name=optimizer_name,
        #     model_parameters=model.parameters(),
        #     lr=learning_rate,
        #     momentum=config.momentum,
        #     weight_decay=config.weight_decay,
        #     nesterov=config.nesterov
        # )

        # scheduler = get_scheduler(
        #     optimizer=optimizer,
        #     scheduler_name=config.scheduler,
        #     t_max=config.get('t_max', 200),
        #     eta_min=config.get('eta_min', 0),
        #     step_size=config.get('step_size', 10),
        #     gamma=config.get('gamma', 0.1),
        #     patience=config.get('scheduler_patience', 10),
        #     factor=config.get('factor', 0.1)
        # )
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=get_device(),
            num_epochs=100,

        )
    finally:
        print("j")
        # wandb.finish()


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


config_file_path = "sweep_config.json"
sweep_config = load_config(config_file_path)
# sweep_id = wandb.sweep(sweep_config, project="training-cifar100-noisy")
# wandb.agent(sweep_id, sweep_train)
sweep_train(sweep_config)


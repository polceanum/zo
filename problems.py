from __future__ import annotations
import os
import gzip
import struct
import urllib.request
import tarfile
import pickle
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================================================================
# Toy problems: Quadratic & a richer test batch
# ================================================================

class QuadraticProblem(nn.Module):
    """
    Axis-aligned quadratic:
        f(x) = 0.5 * sum_i λ_i x_i^2
    with λ_i log-spaced to get condition number `cond`.

    Uses CPU to build eigvals for MPS compatibility.
    """
    def __init__(self, dim=100, cond=10.0, seed=0, device="cpu"):
        super().__init__()
        torch.manual_seed(seed)

        x0 = torch.randn(dim, device=device)
        self.x = nn.Parameter(x0)

        cpu = torch.device("cpu")
        eigvals_cpu = torch.logspace(0, math.log10(cond), dim, device=cpu)
        self.register_buffer("eigvals", eigvals_cpu.to(device))

    def forward(self):
        return 0.5 * torch.sum(self.eigvals * self.x ** 2)


class SphereProblem(nn.Module):
    """
    Simple convex sphere:
        f(x) = 0.5 * ||x||^2
    """
    def __init__(self, dim=100, seed=0, device="cpu"):
        super().__init__()
        torch.manual_seed(seed)
        x0 = torch.randn(dim, device=device)
        self.x = nn.Parameter(x0)

    def forward(self):
        return 0.5 * torch.sum(self.x ** 2)


class RosenbrockProblem(nn.Module):
    """
    Standard multi-dimensional Rosenbrock function:
      f(x) = sum_{i=1}^{n-1} [100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Optimum at x_i = 1.
    """
    def __init__(self, dim=2, seed=0, device="cpu"):
        super().__init__()
        torch.manual_seed(seed)
        x0 = torch.randn(dim, device=device)
        self.x = nn.Parameter(x0)

    def forward(self):
        x = self.x
        return torch.sum(
            100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0
        )


class RastriginProblem(nn.Module):
    """
    Rastrigin function (non-convex, highly multi-modal):
        f(x) = A*n + sum_i [x_i^2 - A cos(2π x_i)]
    Global optimum at x=0 with f(0)=0.
    """
    def __init__(self, dim=20, seed=0, device="cpu", A: float = 10.0):
        super().__init__()
        torch.manual_seed(seed)
        x0 = torch.randn(dim, device=device)
        self.x = nn.Parameter(x0)
        self.A = A

    def forward(self):
        x = self.x
        A = self.A
        return A * x.numel() + torch.sum(x ** 2 - A * torch.cos(2 * math.pi * x))


class AckleyProblem(nn.Module):
    """
    Ackley function (non-convex, "black-box" style benchmark):
        f(x) = -a exp(-b sqrt(1/n sum x_i^2))
               - exp(1/n sum cos(c x_i)) + a + e
    Global optimum at x=0 with f(0)=0.
    """
    def __init__(self, dim=20, seed=0, device="cpu",
                 a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi):
        super().__init__()
        torch.manual_seed(seed)
        x0 = torch.randn(dim, device=device)
        self.x = nn.Parameter(x0)
        self.a = a
        self.b = b
        self.c = c

    def forward(self):
        x = self.x
        n = x.numel()
        a, b, c = self.a, self.b, self.c

        s1 = torch.sum(x ** 2) / n
        s2 = torch.sum(torch.cos(c * x)) / n

        term1 = -a * torch.exp(-b * torch.sqrt(s1 + 1e-12))
        term2 = -torch.exp(s2)
        return term1 + term2 + a + math.e


class L1Problem(nn.Module):
    """
    Non-smooth L1 objective:
        f(x) = sum_i |x_i|
    Gradients are undefined at 0; this is a nice stress test for FO methods,
    while ZO remains well-defined.
    """
    def __init__(self, dim=100, seed=0, device="cpu"):
        super().__init__()
        torch.manual_seed(seed)
        x0 = torch.randn(dim, device=device)
        self.x = nn.Parameter(x0)

    def forward(self):
        return torch.sum(torch.abs(self.x))


class NoisyQuadraticProblem(nn.Module):
    """
    Quadratic + additive noise:
        f(x) = 0.5 * x^T A x + σ * ξ,  ξ ~ N(0,1)
    Mimics a stochastic black-box objective where evaluations are noisy.
    """
    def __init__(self, dim=100, cond=10.0, seed=0,
                 device="cpu", noise_std: float = 0.1):
        super().__init__()
        torch.manual_seed(seed)

        x0 = torch.randn(dim, device=device)
        self.x = nn.Parameter(x0)

        cpu = torch.device("cpu")
        eigvals_cpu = torch.logspace(0, math.log10(cond), dim, device=cpu)
        self.register_buffer("eigvals", eigvals_cpu.to(device))
        self.noise_std = noise_std

    def forward(self):
        quad = 0.5 * torch.sum(self.eigvals * self.x ** 2)
        noise = self.noise_std * torch.randn((), device=self.x.device)
        return quad + noise


# ================================================================
# Minimal MNIST / Fashion-MNIST / CIFAR-10 loaders (no torchvision)
# ================================================================

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

FASHION_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_url(url: str, root: str, filename: str):
    os.makedirs(root, exist_ok=True)
    fpath = os.path.join(root, filename)
    if not os.path.exists(fpath):
        print(f"Downloading {url} -> {fpath}")
        urllib.request.urlretrieve(url, fpath)
    return fpath


def read_idx_images(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8).copy().reshape(num, 1, rows, cols)
    return torch.from_numpy(images)


def read_idx_labels(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8).copy()
    return torch.from_numpy(labels).long()


class MNISTLikeDataset(Dataset):
    def __init__(
        self,
        root: str,
        urls: dict,
        train: bool = True,
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
    ):
        self.root = root
        self.train = train
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        if train:
            img_url = urls["train_images"]
            lbl_url = urls["train_labels"]
        else:
            img_url = urls["test_images"]
            lbl_url = urls["test_labels"]

        img_name = os.path.basename(img_url)
        lbl_name = os.path.basename(lbl_url)

        img_path = download_url(img_url, root, img_name)
        lbl_path = download_url(lbl_url, root, lbl_name)

        images = read_idx_images(img_path)
        labels = read_idx_labels(lbl_path)

        assert images.size(0) == labels.size(0)
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx].float() / 255.0
        y = self.labels[idx]
        x = (x - self.normalize_mean) / self.normalize_std
        return x, y


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 loader (no torchvision), Python version:
    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

    Images: 3x32x32 color
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        normalize_mean=(0.4914, 0.4822, 0.4465),
        normalize_std=(0.2470, 0.2435, 0.2616),
    ):
        self.root = root
        self.train = train

        os.makedirs(root, exist_ok=True)
        archive_name = os.path.basename(CIFAR10_URL)  # cifar-10-python.tar.gz
        archive_path = download_url(CIFAR10_URL, root, archive_name)

        extract_dir = os.path.join(root, "cifar-10-batches-py")
        if not os.path.isdir(extract_dir):
            print(f"Extracting {archive_path} -> {extract_dir}")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=root)

        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 5 + 1)]
        else:
            batch_files = ["test_batch"]

        images_list = []
        labels_list = []

        for bf in batch_files:
            batch_path = os.path.join(extract_dir, bf)
            with open(batch_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data = entry["data"]  # [N, 3072]
            labels = entry["labels"]  # list of ints

            data = data.reshape(-1, 3, 32, 32)  # [N, 3, 32, 32]
            images_list.append(torch.from_numpy(data))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        self.images = torch.cat(images_list, dim=0)
        self.labels = torch.cat(labels_list, dim=0)

        assert self.images.size(0) == self.labels.size(0)

        mean = torch.tensor(normalize_mean).view(3, 1, 1)
        std = torch.tensor(normalize_std).view(3, 1, 1)
        self.normalize_mean = mean
        self.normalize_std = std

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        x = self.images[idx].float() / 255.0  # [3, 32, 32]
        y = self.labels[idx]
        x = (x - self.normalize_mean) / self.normalize_std
        return x, y


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Cifar10CNN(nn.Module):
    """
    Slightly bigger CNN for CIFAR-10.
    Still small enough to run easily on a MacBook Pro.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def get_classification_loaders(
    dataset: str,
    batch_size: int,
    test_batch_size: int,
    use_cuda: bool,
):
    if dataset == "mnist":
        train_ds = MNISTLikeDataset(
            "../data/mnist", MNIST_URLS, train=True,
            normalize_mean=0.1307, normalize_std=0.3081
        )
        test_ds = MNISTLikeDataset(
            "../data/mnist", MNIST_URLS, train=False,
            normalize_mean=0.1307, normalize_std=0.3081
        )
    elif dataset == "fmnist":
        train_ds = MNISTLikeDataset(
            "../data/fmnist", FASHION_URLS, train=True,
            normalize_mean=0.2860, normalize_std=0.3530
        )
        test_ds = MNISTLikeDataset(
            "../data/fmnist", FASHION_URLS, train=False,
            normalize_mean=0.2860, normalize_std=0.3530
        )
    elif dataset == "cifar10":
        train_ds = CIFAR10Dataset("../data/cifar10", train=True)
        test_ds = CIFAR10Dataset("../data/cifar10", train=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_kwargs = {"batch_size": batch_size, "shuffle": True}
    test_kwargs = {"batch_size": test_batch_size, "shuffle": False}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_ds, **train_kwargs)
    test_loader = DataLoader(test_ds, **test_kwargs)
    return train_loader, test_loader


# ================================================================

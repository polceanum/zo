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
from PIL import Image  # type: ignore
import zipfile

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


# ================================================================
# Additional datasets and models: CIFAR-100 and Tiny ImageNet
# ================================================================

class CIFAR100Dataset(Dataset):
    """
    CIFAR-100 dataset loader without torchvision.

    Downloads the CIFAR-100 tarball if not present and extracts images and labels
    from the Python pickled format. Normalizes using ImageNet-like statistics.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        normalize_mean: tuple = (0.5071, 0.4865, 0.4409),
        normalize_std: tuple = (0.2673, 0.2564, 0.2761),
    ) -> None:
        super().__init__()
        self.root = root
        os.makedirs(root, exist_ok=True)
        base_dir = os.path.join(root, "cifar-100-python")
        archive_path = os.path.join(root, "cifar-100-python.tar.gz")

        # Attempt download if extracted folder is missing
        if not os.path.isdir(base_dir):
            url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            try:
                download_url(url, root, "cifar-100-python.tar.gz")
                # Extract tar file
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=root)
            except Exception:
                # If download fails, raise a helpful error to the user
                raise RuntimeError(
                    "CIFAR-100 dataset not found and automatic download failed. "
                    "Please download the archive manually from \n"
                    f"{url} and extract it into {root}."
                )

        # Load data from pickled files
        file_name = "train" if train else "test"
        data_file = os.path.join(base_dir, file_name)
        with open(data_file, "rb") as fo:
            entry = pickle.load(fo, encoding="latin1")
            data = entry["data"]
            labels = entry["fine_labels"]
        data = data.reshape(-1, 3, 32, 32)
        self.images = torch.from_numpy(data)
        self.labels = torch.tensor(labels, dtype=torch.long)
        assert self.images.size(0) == self.labels.size(0)

        mean = torch.tensor(normalize_mean).view(3, 1, 1)
        std = torch.tensor(normalize_std).view(3, 1, 1)
        self.normalize_mean = mean
        self.normalize_std = std

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int):
        x = self.images[idx].float() / 255.0
        y = self.labels[idx]
        x = (x - self.normalize_mean) / self.normalize_std
        return x, y


class Cifar100CNN(nn.Module):
    """
    A simple CNN tailored for CIFAR-100.
    This architecture is a minor extension of the CIFAR-10 model with a larger
    final layer to produce 100 class logits.
    """

    def __init__(self) -> None:
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
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 100),
        )

    def forward(self, x):  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader without torchvision.

    Expects the dataset to be located in the directory structure:
        root/tiny-imagenet-200/
            train/
                <class_id>/images/*.JPEG
            val/
                images/
                val_annotations.txt

    The dataset can be downloaded from:
        http://cs231n.stanford.edu/tiny-imagenet-200.zip

    If the dataset folder is not found, this class will attempt to download and
    extract the zip file. If download fails (e.g., due to network issues), a
    RuntimeError is raised instructing the user to download the dataset manually.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.root = root
        os.makedirs(root, exist_ok=True)
        base_dir = os.path.join(root, "tiny-imagenet-200")
        zip_name = "tiny-imagenet-200.zip"
        zip_path = os.path.join(root, zip_name)

        # If the dataset folder is missing, attempt download
        if not os.path.isdir(base_dir):
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            try:
                download_url(url, root, zip_name)
                # Extract zip file
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(path=root)
            except Exception:
                raise RuntimeError(
                    "Tiny ImageNet dataset not found and automatic download failed. "
                    "Please download the archive manually from \n"
                    f"{url} and extract it into {root}."
                )

        self.image_paths: list[str] = []
        self.labels: list[int] = []
        self.class_to_idx: dict[str, int] = {}

        if train:
            train_dir = os.path.join(base_dir, "train")
            classes = sorted(
                d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
            )
            for idx, cls_name in enumerate(classes):
                self.class_to_idx.setdefault(cls_name, idx)
                images_dir = os.path.join(train_dir, cls_name, "images")
                for fname in os.listdir(images_dir):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    self.image_paths.append(os.path.join(images_dir, fname))
                    self.labels.append(idx)
        else:
            # validation set
            val_dir = os.path.join(base_dir, "val")
            images_dir = os.path.join(val_dir, "images")
            # parse annotations to map filename to class
            anno_path = os.path.join(val_dir, "val_annotations.txt")
            anno_dict: dict[str, str] = {}
            with open(anno_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        file_name, class_id = parts[0], parts[1]
                        anno_dict[file_name] = class_id
                        if class_id not in self.class_to_idx:
                            self.class_to_idx[class_id] = len(self.class_to_idx)
            for fname in os.listdir(images_dir):
                if fname not in anno_dict:
                    continue
                class_id = anno_dict[fname]
                idx = self.class_to_idx[class_id]
                self.image_paths.append(os.path.join(images_dir, fname))
                self.labels.append(idx)

        mean = torch.tensor(normalize_mean).view(3, 1, 1)
        std = torch.tensor(normalize_std).view(3, 1, 1)
        self.normalize_mean = mean
        self.normalize_std = std

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        x = (x - self.normalize_mean) / self.normalize_std
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class TinyImageNetCNN(nn.Module):
    """
    A simple CNN for Tiny ImageNet.
    Uses four convolutional blocks followed by global average pooling and a fully
    connected layer producing logits for 200 classes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 200)

    def forward(self, x):  # type: ignore[override]
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class MLPClassifier(nn.Module):
    """
    Generic multi-layer perceptron for image classification.

    The input images are flattened and passed through a configurable stack of
    fully connected layers with ReLU activations. A final linear layer
    produces class logits which are normalized with `log_softmax`.

    Parameters
    ----------
    in_channels: int
        Number of channels in the input images (e.g. 1 for MNIST, 3 for CIFAR).
    img_size: tuple[int, int]
        Height and width of the input images.
    num_classes: int
        Number of target classes.
    hidden_sizes: list[int], optional
        Sizes of hidden layers. Defaults to [512, 256] if None is provided.
    """
    def __init__(
        self,
        in_channels: int,
        img_size: tuple[int, int],
        num_classes: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        input_dim = in_channels * img_size[0] * img_size[1]
        layers: list[nn.Module] = []
        # First layer flattens and projects
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU(inplace=True))
            last_dim = h
        layers.append(nn.Linear(last_dim, num_classes))
        self.net = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        out = self.net(x)
        return F.log_softmax(out, dim=1)


class VisionTransformerClassifier(nn.Module):
    """
    A minimal Vision Transformer (ViT) for image classification.

    This implementation tokenizes the input image into non-overlapping patches,
    projects each patch into an embedding space, prepends a learnable class
    token, adds positional embeddings, and passes the sequence through a
    Transformer encoder. The output corresponding to the class token is used
    for classification via a linear head.

    Parameters
    ----------
    img_size: tuple[int, int]
        Height and width of the input images.
    patch_size: int
        Size of each square patch. Must divide both height and width evenly.
    in_channels: int
        Number of channels in the input images (e.g. 1 for MNIST, 3 for CIFAR).
    num_classes: int
        Number of target classes.
    embed_dim: int, optional
        Embedding dimension of patch embeddings and transformer. Defaults to 256.
    depth: int, optional
        Number of Transformer encoder layers. Defaults to 4.
    num_heads: int, optional
        Number of attention heads in each Transformer layer. Defaults to 4.
    """
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, (
            "Image dimensions must be divisible by patch_size"
        )
        self.patch_size = patch_size
        self.num_patches = (H // patch_size) * (W // patch_size)
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # type: ignore[override]
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Extract patches using unfold: returns [B, C*patch_size*patch_size, N_patches]
        patches = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        # Transpose to [B, N_patches, patch_dim]
        patches = patches.transpose(1, 2)
        # Embed patches
        tokens = self.patch_embed(patches)
        # Expand class token and concatenate
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # [B, N+1, D]
        # Add positional embeddings
        tokens = tokens + self.pos_embed
        # Transformer encoder
        tokens = self.transformer(tokens)
        # Take class token output
        cls_out = tokens[:, 0]
        cls_out = self.norm(cls_out)
        out = self.head(cls_out)
        return F.log_softmax(out, dim=1)


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
    elif dataset == "cifar100":
        train_ds = CIFAR100Dataset("../data/cifar100", train=True)
        test_ds = CIFAR100Dataset("../data/cifar100", train=False)
    elif dataset in ["tinyimagenet", "tiny-imagenet", "tiny_imagenet", "tiny-imagenet-200", "tinynet"]:
        # Accept several aliases for Tiny ImageNet
        train_ds = TinyImageNetDataset("../data/tinyimagenet", train=True)
        test_ds = TinyImageNetDataset("../data/tinyimagenet", train=False)
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

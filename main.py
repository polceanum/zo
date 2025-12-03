from __future__ import annotations
import argparse
import math
import os
import time
import gzip
import struct
import urllib.request
import random
import json
import hashlib
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tarfile
import pickle

from mezo import MeZO  # custom ZO optimizer


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
# Optimizer registry
# ================================================================

def build_optimizer_registry(zo_eps: float) -> Dict[str, Callable]:
    def mezo_factory(variant: str, weight_decay: float = 0.0):
        def _f(params, lr):
            return MeZO(
                params,
                lr=lr,
                epsilon=zo_eps,
                variant=variant,
                weight_decay=weight_decay,
            )
        return _f

    registry = {
        "sgd":                lambda params, lr: torch.optim.SGD(params, lr=lr),
        "sgd_momentum":       lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
        "adam":               lambda params, lr: torch.optim.Adam(params, lr=lr),
        "adamw":              lambda params, lr: torch.optim.AdamW(params, lr=lr),
        "rmsprop":            lambda params, lr: torch.optim.RMSprop(params, lr=lr),
        "adagrad":            lambda params, lr: torch.optim.Adagrad(params, lr=lr),
        "mezo_sgd":           mezo_factory("sgd"),
        "mezo_adam":          mezo_factory("adam"),
        "mezo_adamw":         mezo_factory("adamw"),
        "mezo_adam_adapt":    mezo_factory("adam_adapt"),
        "mezo_adam_adapt2":   mezo_factory("adam_adapt2"),
        "mezo_adam_adapt3":   mezo_factory("adam_adapt3"),
    }
    return registry


# ================================================================
# Result caching helpers
# ================================================================

def load_results_db(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r") as f:
            db = json.load(f)
        if "runs" not in db:
            db = {"schema_version": 1, "runs": {}}
    else:
        db = {"schema_version": 1, "runs": {}}
    return db


def save_results_db(path: str, db: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp_path, path)


def make_run_config(
    kind: str,
    problem: str,
    optimizer_name: str,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = {
        "kind": kind,
        "problem": problem,
        "optimizer": optimizer_name,
        "seed": seed,
        "device": str(device),
        "zo_eps": args.zo_eps,
    }
    cfg.update(extra)
    return cfg


def run_id_from_config(cfg: Dict[str, Any]) -> str:
    cfg_json = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()


# ================================================================
# W&B helper (optional)
# ================================================================

def init_wandb(args: argparse.Namespace):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb logging requested but `wandb` is not installed. "
              "Run `pip install wandb` to enable it.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config=vars(args),
    )
    return run


def wandb_log(run, metrics: Dict[str, Any]):
    if run is None:
        return
    import wandb  # type: ignore
    wandb.log(metrics)


# ================================================================
# Training/eval loops
# ================================================================

def run_toy_problem(
    problem_name: str,
    opt_name: str,
    opt_fn: Callable,
    dim: int,
    steps: int,
    lr: float,
    device: torch.device,
    seed: int,
    cond: float,
    wandb_run=None,
    record_traj: bool = False,
    traj_dir: str = "toy_trajectories",
    run_id: Optional[str] = None,
) -> Dict:
    torch.manual_seed(seed)

    if problem_name == "quadratic":
        model = QuadraticProblem(dim=dim, cond=cond, seed=seed, device=device).to(device)
    elif problem_name == "sphere":
        model = SphereProblem(dim=dim, seed=seed, device=device).to(device)
    elif problem_name == "rosenbrock":
        model = RosenbrockProblem(dim=min(dim, 10), seed=seed, device=device).to(device)
    elif problem_name == "rastrigin":
        dim_eff = min(dim, 50)
        model = RastriginProblem(dim=dim_eff, seed=seed, device=device).to(device)
    elif problem_name == "ackley":
        dim_eff = min(dim, 50)
        model = AckleyProblem(dim=dim_eff, seed=seed, device=device).to(device)
    elif problem_name == "l1":
        model = L1Problem(dim=dim, seed=seed, device=device).to(device)
    elif problem_name == "noisy_quadratic":
        model = NoisyQuadraticProblem(dim=dim, cond=cond, seed=seed, device=device).to(device)
    else:
        raise ValueError(f"Unknown toy problem: {problem_name}")

    optimizer = opt_fn(model.parameters(), lr=lr)
    from mezo import MeZO as MeZOClass  # avoid name shadow in type-checkers
    is_zo = isinstance(optimizer, MeZOClass)

    # record full parameter vector trajectory if requested and model has x
    traj: Optional[List[np.ndarray]] = None
    if record_traj and hasattr(model, "x"):
        traj = []

    start_time = time.time()
    best_loss = float("inf")
    losses = []

    for step in range(steps):
        if is_zo:
            def closure():
                return model()
            loss_val = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = model()
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())

        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        if traj is not None:
            x_np = model.x.detach().cpu().numpy().copy()
            traj.append(x_np)

        if wandb_run is not None and (step + 1) % max(steps // 10, 1) == 0:
            wandb_log(
                wandb_run,
                {
                    "kind": "toy",
                    "toy_problem": problem_name,
                    "optimizer": opt_name,
                    "step": step + 1,
                    "toy_loss": loss_val,
                },
            )

    elapsed = time.time() - start_time

    traj_path = None
    if traj is not None and len(traj) > 0:
        os.makedirs(traj_dir, exist_ok=True)
        traj_arr = np.stack(traj, axis=0)  # [steps, dim]
        base_name = f"{problem_name}_{opt_name}_seed{seed}"
        if run_id is not None:
            base_name = f"{problem_name}_{opt_name}_seed{seed}_{run_id[:8]}"

        traj_path = os.path.join(traj_dir, base_name + "_traj.npy")
        meta_path = os.path.join(traj_dir, base_name + "_meta.json")

        np.save(traj_path, traj_arr)
        meta = {
            "problem": problem_name,
            "optimizer": opt_name,
            "seed": seed,
            "dim": int(traj_arr.shape[1]),
            "steps": int(traj_arr.shape[0]),
            "run_id": run_id,
            "traj_path": os.path.abspath(traj_path),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    result = {
        "kind": "toy",
        "problem": problem_name,
        "optimizer": opt_name,
        "dim": dim,
        "steps": steps,
        "seed": seed,
        "final_loss": losses[-1],
        "best_loss": best_loss,
        "time_sec": elapsed,
        "traj_path": traj_path,
    }
    print(
        f"-> FinalLoss={result['final_loss']:.6e} "
        f"BestLoss={result['best_loss']:.6e} "
        f"Time={result['time_sec']:.2f}s"
    )
    if traj_path is not None:
        print(f"   Trajectory saved to: {traj_path}")
    return result


def train_epoch_classification(
    model: nn.Module,
    optimizer,
    train_loader,
    device: torch.device,
    is_zo: bool,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for data, target in train_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if is_zo:
            def closure():
                output = model(data)
                return F.nll_loss(output, target)

            loss_val = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())

        running_loss += loss_val
        num_batches += 1

    return running_loss / max(1, num_batches)


@torch.no_grad()
def test_classification(
    model: nn.Module,
    test_loader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum").item()
        test_loss += loss
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    test_loss /= total
    acc = 100.0 * correct / total
    return {"loss": test_loss, "accuracy": acc}


def run_classification_problem(
    dataset: str,
    opt_name: str,
    opt_fn: Callable,
    epochs: int,
    lr: float,
    batch_size: int,
    test_batch_size: int,
    device: torch.device,
    use_cuda: bool,
    seed: int,
    wandb_run=None,
) -> Dict:
    torch.manual_seed(seed)

    train_loader, test_loader = get_classification_loaders(
        dataset, batch_size, test_batch_size, use_cuda=use_cuda
    )

    if dataset in ("mnist", "fmnist"):
        model = SimpleCNN().to(device)
    elif dataset == "cifar10":
        model = Cifar10CNN().to(device)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    optimizer = opt_fn(model.parameters(), lr=lr)
    from mezo import MeZO as MeZOClass  # avoid confusion in isinstance
    is_zo = isinstance(optimizer, MeZOClass)

    start_time = time.time()
    best_acc = 0.0
    best_test_loss = float("inf")
    last_test_stats = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_classification(model, optimizer, train_loader, device, is_zo)
        test_stats = test_classification(model, test_loader, device)
        last_test_stats = test_stats

        if test_stats["accuracy"] > best_acc:
            best_acc = test_stats["accuracy"]
        if test_stats["loss"] < best_test_loss:
            best_test_loss = test_stats["loss"]

        print(
            f"[{dataset}] Opt={opt_name} Epoch={epoch}/{epochs} "
            f"TrainLoss={train_loss:.4f} "
            f"TestLoss={test_stats['loss']:.4f} "
            f"TestAcc={test_stats['accuracy']:.2f}%"
        )

        if wandb_run is not None:
            wandb_log(
                wandb_run,
                {
                    "kind": "ml",
                    "dataset": dataset,
                    "optimizer": opt_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_stats["loss"],
                    "test_acc": test_stats["accuracy"],
                },
            )

    elapsed = time.time() - start_time

    result = {
        "kind": "ml",
        "problem": dataset,
        "optimizer": opt_name,
        "epochs": epochs,
        "seed": seed,
        "final_test_loss": last_test_stats["loss"],
        "final_test_acc": last_test_stats["accuracy"],
        "best_test_loss": best_test_loss,
        "best_test_acc": best_acc,
        "time_sec": elapsed,
    }
    print(
        f"-> FinalTestLoss={result['final_test_loss']:.4f} "
        f"FinalTestAcc={result['final_test_acc']:.2f}% "
        f"BestAcc={result['best_test_acc']:.2f}% "
        f"Time={result['time_sec']:.2f}s"
    )
    return result


# ================================================================
# Pretty-print summaries
# ================================================================

def print_summaries(all_results: List[Dict]):
    by_problem: Dict[tuple, List[Dict]] = {}
    for r in all_results:
        key = (r["kind"], r["problem"])
        by_problem.setdefault(key, []).append(r)

    print("\n" + "=" * 72)
    print("SUMMARY BY TASK")
    print("=" * 72)

    for (kind, problem), results in sorted(by_problem.items()):
        print()
        if kind == "toy":
            header = f"[TOY] {problem}"
            print(header)
            print("-" * len(header))
            print(f"{'optimizer':<14} {'final_loss':>14} {'best_loss':>14} {'time[s]':>10}")
            print("-" * 56)
            for r in sorted(results, key=lambda x: x["optimizer"]):
                print(
                    f"{r['optimizer']:<14} "
                    f"{r['final_loss']:>14.3e} "
                    f"{r['best_loss']:>14.3e} "
                    f"{r['time_sec']:>10.2f}"
                )
        else:
            header = f"[ML] {problem}"
            print(header)
            print("-" * len(header))
            print(f"{'optimizer':<14} {'final_acc[%]':>14} {'best_acc[%]':>14} "
                  f"{'final_loss':>14} {'time[s]':>10}")
            print("-" * 74)
            for r in sorted(results, key=lambda x: x["optimizer"]):
                print(
                    f"{r['optimizer']:<14} "
                    f"{r['final_test_acc']:>14.2f} "
                    f"{r['best_test_acc']:>14.2f} "
                    f"{r['final_test_loss']:>14.4f} "
                    f"{r['time_sec']:>10.2f}"
                )


# ================================================================
# Main harness / CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimizer benchmark harness (no torchvision, MeZO external, cached + W&B)"
    )

    parser.add_argument("--device", type=str, default="auto",
                        help="cpu | cuda | mps | auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1],
                        help="Random seeds for repeated runs")
    parser.add_argument("--zo-eps", type=float, default=1e-3,
                        help="Epsilon for zeroth-order methods")

    parser.add_argument(
        "--toy-problems",
        type=str,
        nargs="*",
        default=[
            "quadratic",
            "sphere",
            "rosenbrock",
            "rastrigin",
            "ackley",
            "l1",
            "noisy_quadratic",
        ],
        help="Toy problems: quadratic, sphere, rosenbrock, rastrigin, ackley, l1, noisy_quadratic",
    )
    parser.add_argument("--toy-steps", type=int, default=2000,
                        help="Steps for toy problems")
    parser.add_argument("--toy-dim", type=int, default=100,
                        help="Dimensionality for toy problems")
    parser.add_argument("--toy-lr", type=float, default=1e-2,
                        help="Learning rate for toy problems")
    parser.add_argument("--toy-cond", type=float, default=10.0,
                        help="Condition number for quadratic/noisy_quadratic problem")

    parser.add_argument(
        "--ml-problems",
        type=str,
        nargs="*",
        default=["mnist", "fmnist", "cifar10"],
        help="ML problems: mnist, fmnist, cifar10",
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epochs for ML problems")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--ml-lr", type=float, default=1e-3,
                        help="Learning rate for ML problems")

    parser.add_argument("--optimizers", type=str, nargs="*",
                        default=[
                            "sgd",
                            "sgd_momentum",
                            "adam",
                            "adamw",
                            "rmsprop",
                            "adagrad",
                            "mezo_sgd",
                            "mezo_adam",
                            "mezo_adamw",
                            "mezo_adam_adapt",
                            "mezo_adam_adapt2",
                            "mezo_adam_adapt3",
                        ],
                        help="Optimizers to benchmark")

    parser.add_argument("--results-path", type=str, default="results.json",
                        help="Path to JSON file for cached results")

    # Toy trajectory saving
    parser.add_argument("--save-toy-trajectories", action="store_true", default=False,
                        help="If set, save parameter trajectories for toy problems")
    parser.add_argument("--toy-traj-dir", type=str, default="toy_trajectories",
                        help="Directory to save toy trajectories")

    # W&B options
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="optimizer-bench",
                        help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="wandb entity / team (optional)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="wandb run name (optional)")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    print("Using device:", device)

    opt_registry = build_optimizer_registry(args.zo_eps)

    optimizers_to_run = []
    for name in args.optimizers:
        if name not in opt_registry:
            raise ValueError(f"Unknown optimizer: {name}")
        optimizers_to_run.append(name)

    results_db = load_results_db(args.results_path)
    runs_store = results_db["runs"]
    wandb_run = init_wandb(args)

    all_results: List[Dict] = []

    # ----------------- Toy problems -----------------
    for seed in args.seeds:
        for problem_name in args.toy_problems:
            dim = args.toy_dim

            for opt_name in optimizers_to_run:
                opt_fn = opt_registry[opt_name]

                config = make_run_config(
                    kind="toy",
                    problem=problem_name,
                    optimizer_name=opt_name,
                    seed=seed,
                    device=device,
                    args=args,
                    extra={
                        "dim": dim,
                        "steps": args.toy_steps,
                        "toy_lr": args.toy_lr,
                        "toy_cond": args.toy_cond,
                    },
                )
                run_id = run_id_from_config(config)

                if run_id in runs_store:
                    print(f"\n[TOY] Problem={problem_name} Opt={opt_name} Seed={seed} "
                          f"=> SKIP (cached in {args.results_path})")
                    cached = runs_store[run_id]["metrics"]
                    all_results.append(cached)
                    continue

                print(f"\n[TOY] Problem={problem_name} Opt={opt_name} Seed={seed}")
                res = run_toy_problem(
                    problem_name=problem_name,
                    opt_name=opt_name,
                    opt_fn=opt_fn,
                    dim=dim,
                    steps=args.toy_steps,
                    lr=args.toy_lr,
                    device=device,
                    seed=seed,
                    cond=args.toy_cond,
                    wandb_run=wandb_run,
                    record_traj=args.save_toy_trajectories,
                    traj_dir=args.toy_traj_dir,
                    run_id=run_id,
                )
                all_results.append(res)
                runs_store[run_id] = {"config": config, "metrics": res}
                save_results_db(args.results_path, results_db)

    # ----------------- ML problems ------------------
    for seed in args.seeds:
        for dataset in args.ml_problems:
            for opt_name in optimizers_to_run:
                opt_fn = opt_registry[opt_name]

                config = make_run_config(
                    kind="ml",
                    problem=dataset,
                    optimizer_name=opt_name,
                    seed=seed,
                    device=device,
                    args=args,
                    extra={
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "test_batch_size": args.test_batch_size,
                        "ml_lr": args.ml_lr,
                    },
                )
                run_id = run_id_from_config(config)

                if run_id in runs_store:
                    print(f"\n[ML] Dataset={dataset} Opt={opt_name} Seed={seed} "
                          f"=> SKIP (cached in {args.results_path})")
                    cached = runs_store[run_id]["metrics"]
                    all_results.append(cached)
                    continue

                print(f"\n[ML] Dataset={dataset} Opt={opt_name} Seed={seed}")
                res = run_classification_problem(
                    dataset=dataset,
                    opt_name=opt_name,
                    opt_fn=opt_fn,
                    epochs=args.epochs,
                    lr=args.ml_lr,
                    batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size,
                    device=device,
                    use_cuda=use_cuda,
                    seed=seed,
                    wandb_run=wandb_run,
                )
                all_results.append(res)
                runs_store[run_id] = {"config": config, "metrics": res}
                save_results_db(args.results_path, results_db)

    print_summaries(all_results)

    if wandb_run is not None:
        import wandb  # type: ignore
        wandb_run.finish()


if __name__ == "__main__":
    main()

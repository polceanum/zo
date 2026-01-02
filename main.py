from __future__ import annotations
import argparse
import math
import os
import time
import json
import hashlib
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mezo import MeZO  # custom ZO optimizer
from problems import (
    QuadraticProblem,
    SphereProblem,
    RosenbrockProblem,
    RastriginProblem,
    AckleyProblem,
    L1Problem,
    NoisyQuadraticProblem,
    SimpleCNN,
    Cifar10CNN,
    get_classification_loaders,
)



# ================================================================
# Problems are defined in problems.py
# ================================================================

# ================================================================
# BERT on Wikitext-2 MLM (Wikipedia-ish)
# ================================================================

def run_bert_wiki_mlm_problem(
    dataset: str,
    opt_name: str,
    opt_fn: Callable,
    epochs: int,
    lr: float,               # this is the BERT LR (usually ~1e-5–5e-5)
    batch_size: int,
    test_batch_size: int,
    device: torch.device,
    use_cuda: bool,
    seed: int,
    wandb_run=None,
) -> Dict:
    """
    BERT-base masked LM on Wikitext-2 (Wikipedia-ish).
    Uses HuggingFace `datasets` + `transformers` with standard MLM setup.

    For stability we use a BERT-specific LR (passed in as `lr`) and guard
    against non-finite losses (both FO and ZO).
    """
    torch.manual_seed(seed)

    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
    )

    print("Loading Wikitext-2 dataset for BERT MLM...")
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Slightly shorter seq length to save memory but still realistic
    max_length = 64

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_ds = tokenized["train"]
    eval_ds = tokenized["validation"]

    # Keep runtime reasonable by sub-sampling
    max_train_examples = 50000
    max_eval_examples = 10000
    if len(train_ds) > max_train_examples:
        train_ds = train_ds.select(range(max_train_examples))
    if len(eval_ds) > max_eval_examples:
        eval_ds = eval_ds.select(range(max_eval_examples))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # ---- cap effective BERT batch sizes to avoid OOM ----
    bert_train_batch_size = min(batch_size, 8)
    bert_eval_batch_size = min(test_batch_size, 16)

    print(
        f"Using BERT batch sizes: train={bert_train_batch_size}, "
        f"eval={bert_eval_batch_size}, max_length={max_length}, lr={lr:g}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=bert_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=bert_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    print("Loading BERT-base-uncased for MLM...")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    # Some light memory tweaks (common in HF training)
    model.resize_token_embeddings(len(tokenizer))  # in case tokenizer changed
    model.config.use_cache = False  # disable cache for training
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        # Older transformers versions may not have this
        pass

    model.to(device)

    optimizer = opt_fn(model.parameters(), lr=lr)
    from mezo import MeZO as MeZOClass
    is_zo = isinstance(optimizer, MeZOClass)

    def evaluate() -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        print(f"[{dataset}] Opt={opt_name} Running evaluation...", flush=True)

        with torch.no_grad():
            total_eval_batches = len(eval_loader)
            for batch_idx, batch in enumerate(eval_loader, start=1):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = batch["labels"]
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    token_type_ids=batch.get("token_type_ids", None),
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits  # [B, L, V]

                mask = labels != -100
                if mask.any():
                    preds = logits.argmax(dim=-1)
                    correct = (preds[mask] == labels[mask]).sum().item()
                    count = mask.sum().item()
                    total_correct += correct
                    total_count += count

                num_tokens = mask.sum().item()
                if num_tokens == 0:
                    num_tokens = 1
                total_loss += float(loss.item()) * num_tokens

                # light eval progress every ~10% of eval
                if total_eval_batches > 0:
                    log_every_eval = max(1, total_eval_batches // 10)
                    if batch_idx % log_every_eval == 0 or batch_idx == total_eval_batches:
                        print(
                            f"[{dataset}] Opt={opt_name} Eval batch "
                            f"{batch_idx}/{total_eval_batches}",
                            flush=True,
                        )

        avg_loss = total_loss / max(1, total_count)
        acc = 100.0 * total_correct / max(1, total_count) if total_count > 0 else 0.0
        return {"loss": avg_loss, "accuracy": acc}

    start_time = time.time()
    best_acc = 0.0
    best_eval_loss = float("inf")
    last_eval_stats: Optional[Dict[str, float]] = None

    total_batches = len(train_loader)
    log_every = max(1, total_batches // 10)  # ~10 updates per epoch

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["labels"]

            if is_zo:
                # ----------------- ZO path with non-finite guards -----------------
                def closure():
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask", None),
                        token_type_ids=batch.get("token_type_ids", None),
                        labels=labels,
                    )
                    loss = out.loss
                    if not torch.isfinite(loss):
                        print(
                            f"[{dataset}] Opt={opt_name} Epoch={epoch} "
                            f"Batch={batch_idx}/{total_batches} "
                            f"Non-finite ZO loss={loss.item()}, returning 0.",
                            flush=True,
                        )
                        # Return a benign scalar so MeZO sees zero gradient
                        return torch.zeros((), device=device, dtype=loss.dtype)
                    return loss

                loss_val = optimizer.step(closure)
                # MeZO.step now has its own non-finite guard, but we double-check.
                if (loss_val is None) or (not math.isfinite(loss_val)):
                    print(
                        f"[{dataset}] Opt={opt_name} Epoch={epoch} "
                        f"Batch={batch_idx}/{total_batches} "
                        f"MeZO step returned non-finite loss={loss_val}, "
                        f"skipping logging for this batch.",
                        flush=True,
                    )
                    continue

            else:
                # ----------------- FO path (already guarded) -----------------
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    token_type_ids=batch.get("token_type_ids", None),
                    labels=labels,
                )
                loss = out.loss

                # ---- non-finite loss guard ----
                if not torch.isfinite(loss):
                    print(
                        f"[{dataset}] Opt={opt_name} Epoch={epoch} "
                        f"Batch={batch_idx}/{total_batches} "
                        f"Non-finite loss={loss.item()}, skipping batch.",
                        flush=True,
                    )
                    optimizer.zero_grad()
                    continue

                loss.backward()
                # optional: gradient clipping if you want extra safety
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_val = float(loss.item())

            running_loss += loss_val
            num_batches += 1

            # ---- progress print every ~10% of the epoch ----
            if batch_idx % log_every == 0 or batch_idx == total_batches:
                avg_batch_loss = running_loss / max(1, num_batches)
                print(
                    f"[{dataset}] Opt={opt_name} Epoch={epoch}/{epochs} "
                    f"Batch={batch_idx}/{total_batches} "
                    f"RunningTrainLoss={avg_batch_loss:.4f}",
                    flush=True,
                )

        train_loss = running_loss / max(1, num_batches) if num_batches > 0 else float("nan")
        eval_stats = evaluate()
        last_eval_stats = eval_stats

        if eval_stats["accuracy"] > best_acc:
            best_acc = eval_stats["accuracy"]
        if eval_stats["loss"] < best_eval_loss:
            best_eval_loss = eval_stats["loss"]

        print(
            f"[{dataset}] Opt={opt_name} Epoch={epoch}/{epochs} "
            f"TrainLoss={train_loss:.4f} "
            f"EvalLoss={eval_stats['loss']:.4f} "
            f"EvalAcc={eval_stats['accuracy']:.2f}%",
            flush=True,
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
                    "test_loss": eval_stats["loss"],
                    "test_acc": eval_stats["accuracy"],
                },
            )

    elapsed = time.time() - start_time

    result = {
        "kind": "ml",
        "problem": dataset,
        "optimizer": opt_name,
        "epochs": epochs,
        "seed": seed,
        "final_test_loss": last_eval_stats["loss"] if last_eval_stats is not None else float("nan"),
        "final_test_acc": last_eval_stats["accuracy"] if last_eval_stats is not None else 0.0,
        "best_test_loss": best_eval_loss,
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
# Optimizer registry
# ================================================================

def build_optimizer_registry(zo_eps: float) -> Dict[str, Callable]:
    """Return a mapping from optimizer name -> factory(params, lr).

    Includes first-order baselines, MeZO baselines, and experimental variants:
      - mezo_sgd
      - mezo_sgd_adapt
      - mezo_adamu
    """

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

    registry: Dict[str, Callable] = {
        # First-order baselines
        "sgd":          lambda params, lr: torch.optim.SGD(params, lr=lr),
        "sgd_momentum": lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
        "adam":         lambda params, lr: torch.optim.Adam(params, lr=lr),
        "adamw":        lambda params, lr: torch.optim.AdamW(params, lr=lr),

        # Zeroth-order baselines (MeZO)
        "mezo_sgd":     mezo_factory("sgd"),
        "mezo_adamu":   mezo_factory("adamu"),

        # Zeroth-order experimental variants
        "mezo_sgd_adapt":     mezo_factory("sgd_adapt"),
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
# Training/eval loops (toy + image classification)
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
            # ZO optimizers may return a Python float or a Tensor; ignore the return value for logging.
            def closure():
                return model()
            _ = optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = model()
            loss.backward()
            optimizer.step()

        # Always log the true objective f(theta) at the *current* parameters (comparable across optimizers).
        with torch.no_grad():
            loss_val = float(model().item())

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
            f"TestAcc={test_stats['accuracy']:.2f}%",
            flush=True,
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
                    f"{r.get('final_test_acc', 0.0):>14.2f} "
                    f"{r.get('best_test_acc', 0.0):>14.2f} "
                    f"{r.get('final_test_loss', 0.0):>14.4f} "
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
    parser.add_argument("--toy-lr", type=float, default=None,
                        help="Learning rate for toy problems")
    parser.add_argument("--toy-cond", type=float, default=10.0,
                        help="Condition number for quadratic/noisy_quadratic problem")

    parser.add_argument(
        "--ml-problems",
        type=str,
        nargs="*",
        default=["mnist", "fmnist", "cifar10"], # "bert_wiki_mlm"
        help="ML problems: mnist, fmnist, cifar10, bert_wiki_mlm",
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epochs for ML problems")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--ml-lr", type=float, default=None,
                        help="Learning rate for non-BERT ML problems")
    parser.add_argument("--bert-lr", type=float, default=5e-5,
                        help="Learning rate for BERT MLM problems")

    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="*",
        default=[
            "sgd",
            "sgd_momentum",
            "adam",
            "adamw",
            "mezo_sgd",
            "mezo_adamu",
            "mezo_sgd_adapt",
        ],
        help="Optimizers to benchmark",
    )

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

    # ------------------------------------------------------------
    # Hyperparameter defaults (only used when the corresponding CLI flag is omitted)
    # ------------------------------------------------------------
    def default_toy_lr(opt_name: str) -> float:
        # Toy problems are small and deterministic, so higher LR is often fine.
        if opt_name in ("adam", "adamw"):
            return 1e-2
        if opt_name.startswith("mezo_"):
            return 1e-2
        # SGD / SGD+momentum
        return 1e-2

    def default_ml_lr(problem_name: str, opt_name: str) -> float:
        # Reasonable, widely-used starting points for the small CNN/MLP baselines.
        if opt_name in ["adam", "adamw"]:
            return 1e-3
        if opt_name in ["mezo_sgd", "mezo_adamu"]:
            return 1e-4
        if opt_name in ["mezo_sgd_adapt"]:
            return 1e-3
        # SGD family: higher LR typically needed for CIFAR-10 vs MNIST.
        return 1e-1 if problem_name == "cifar10" else 5e-2


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
                        "toy_lr": (args.toy_lr if args.toy_lr is not None else default_toy_lr(opt_name)),
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
                    lr=(args.toy_lr if args.toy_lr is not None else default_toy_lr(opt_name)),
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

                if dataset == "bert_wiki_mlm":
                    extra_cfg = {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "test_batch_size": args.test_batch_size,
                        "bert_lr": args.bert_lr,
                    }
                else:
                    extra_cfg = {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "test_batch_size": args.test_batch_size,
                        "ml_lr": (args.ml_lr if args.ml_lr is not None else default_ml_lr(problem_name, opt_name)),
                    }

                config = make_run_config(
                    kind="ml",
                    problem=dataset,
                    optimizer_name=opt_name,
                    seed=seed,
                    device=device,
                    args=args,
                    extra=extra_cfg,
                )
                run_id = run_id_from_config(config)

                if run_id in runs_store:
                    print(f"\n[ML] Dataset={dataset} Opt={opt_name} Seed={seed} "
                          f"=> SKIP (cached in {args.results_path})")
                    cached = runs_store[run_id]["metrics"]
                    all_results.append(cached)
                    continue

                print(f"\n[ML] Dataset={dataset} Opt={opt_name} Seed={seed}")
                if dataset == "bert_wiki_mlm":
                    res = run_bert_wiki_mlm_problem(
                        dataset=dataset,
                        opt_name=opt_name,
                        opt_fn=opt_fn,
                        epochs=args.epochs,
                        lr=args.bert_lr,
                        batch_size=args.batch_size,
                        test_batch_size=args.test_batch_size,
                        device=device,
                        use_cuda=use_cuda,
                        seed=seed,
                        wandb_run=wandb_run,
                    )
                else:
                    res = run_classification_problem(
                        dataset=dataset,
                        opt_name=opt_name,
                        opt_fn=opt_fn,
                        epochs=args.epochs,
                        lr=(args.ml_lr if args.ml_lr is not None else default_ml_lr(problem_name, opt_name)),
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

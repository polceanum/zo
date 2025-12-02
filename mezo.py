from __future__ import annotations
import math
import random
from typing import Dict, Callable, Any

import torch
from torch.optim import Optimizer


class MeZO(Optimizer):
    r"""
    Memory-efficient zeroth-order optimizer (MeZO-style).

    Uses a 2-point SPSA-like rank-1 gradient estimate:
        ĝ ≈ ((L(θ + εz) - L(θ - εz)) / (2ε)) * z

    Variants:
        - 'sgd'   : ZO-SGD
        - 'adam'  : ZO-Adam (coupled weight decay)
        - 'adamw' : ZO-AdamW (decoupled weight decay)

    Key design choices:
      - Does NOT assume anything about the model (it may use dropout etc.).
      - Controls RNG *inside* step() so that f(θ+εz) and f(θ−εz) see
        identical stochasticity.
      - Uses a separate per-device Generator for z (noise directions),
        so model RNG and direction RNG are independent.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        epsilon: float = 1e-3,
        variant: str = "sgd",   # 'sgd', 'adam', 'adamw'
        betas=(0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        projected_grad_clip: float | None = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if variant not in ("sgd", "adam", "adamw"):
            raise ValueError(f"Unknown variant: {variant}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            epsilon=epsilon,
            variant=variant,
            betas=betas,
            adam_eps=adam_eps,
            weight_decay=weight_decay,
            projected_grad_clip=projected_grad_clip,
        )
        super().__init__(params, defaults)

        # cache per-device generators for z to avoid reallocation
        self._gens: Dict[torch.device, torch.Generator] = {}

    # ---------------- RNG helpers ----------------

    def _get_gen(self, device: torch.device) -> torch.Generator:
        g = self._gens.get(device)
        if g is None:
            g = torch.Generator(device=device)
            self._gens[device] = g
        return g

    def _save_rng_state(self, device: torch.device) -> Dict[str, Any]:
        """
        Save global RNG state for CPU and the given device (cuda/mps),
        so we can replay identical stochasticity for f(θ+εz) and f(θ−εz).
        """
        state: Dict[str, Any] = {}
        state["cpu"] = torch.get_rng_state()

        if device.type == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state(device)
        elif device.type == "mps" and hasattr(torch, "mps"):
            # MPS RNG API mirrors CUDA's
            try:
                state["mps"] = torch.mps.get_rng_state()
            except Exception:
                pass

        return state

    def _restore_rng_state(self, device: torch.device, state: Dict[str, Any]) -> None:
        """
        Restore previously saved RNG state.
        """
        if "cpu" in state:
            torch.set_rng_state(state["cpu"])

        if device.type == "cuda" and "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda"], device)
        elif device.type == "mps" and "mps" in state and hasattr(torch, "mps"):
            try:
                torch.mps.set_rng_state(state["mps"])
            except Exception:
                pass

    # ---------------- main step() ----------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        if closure is None:
            raise RuntimeError("MeZO.step() requires a closure returning the loss.")

        # find first param for device
        first_param = None
        for group in self.param_groups:
            if group["params"]:
                first_param = group["params"][0]
                if first_param is not None:
                    break
        if first_param is None:
            return None

        device = first_param.device

        # separate RNG for direction sampling (z)
        gen = self._get_gen(device)
        # one random seed per step for z (does NOT affect model RNG)
        seed = random.randrange(2**31 - 1)
        gen.manual_seed(seed)

        main_group = self.param_groups[0]
        eps = main_group["epsilon"]
        proj_clip = main_group.get("projected_grad_clip", None)

        # save global RNG so we can replay for both f+ and f-
        rng_state = self._save_rng_state(device)

        # --------------------------------------------------
        # Loop 1: sample z and apply θ <- θ + εz
        # --------------------------------------------------
        for group in self.param_groups:
            epsilon = group["epsilon"]
            params = group["params"]
            if not params:
                continue
            for p in params:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                # persistent noise buffer
                if "z" not in state:
                    state["z"] = torch.empty_like(p)
                z = state["z"]
                z.normal_(generator=gen)  # in-place sample
                p.add_(epsilon * z)

        # f(θ + εz) with restored RNG
        self._restore_rng_state(device, rng_state)
        loss_plus = closure()
        loss_plus_val = float(loss_plus.detach().item())

        # --------------------------------------------------
        # Loop 2: apply θ <- θ - 2εz (now θ = θ - εz)
        # --------------------------------------------------
        for group in self.param_groups:
            epsilon = group["epsilon"]
            params = group["params"]
            if not params:
                continue
            for p in params:
                if not p.requires_grad:
                    continue
                z = self.state[p]["z"]
                p.add_(-2.0 * epsilon * z)

        # f(θ - εz) with the *same* RNG stream
        self._restore_rng_state(device, rng_state)
        loss_minus = closure()
        loss_minus_val = float(loss_minus.detach().item())

        # scalar projected gradient
        projected_grad = (loss_plus_val - loss_minus_val) / (2.0 * eps)
        if proj_clip is not None:
            if projected_grad > proj_clip:
                projected_grad = proj_clip
            elif projected_grad < -proj_clip:
                projected_grad = -proj_clip

        # --------------------------------------------------
        # Loop 3: restore θ and apply SGD/Adam/AdamW update
        # --------------------------------------------------
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            adam_eps = group["adam_eps"]
            weight_decay = group["weight_decay"]
            epsilon = group["epsilon"]
            params = group["params"]
            if not params:
                continue

            decoupled_wd = (group["variant"] == "adamw")
            use_adam = group["variant"] in ("adam", "adamw")

            for p in params:
                if not p.requires_grad:
                    continue

                state = self.state[p]
                z = state["z"]

                # restore parameters: θ <- θ + εz (back to original θ)
                p.add_(epsilon * z)

                if group["variant"] == "sgd":
                    # ZO-SGD: θ <- θ - lr * (proj_grad * z [+ wd])
                    gparam = projected_grad * z
                    if weight_decay != 0.0:
                        gparam.add_(p.data, alpha=weight_decay)
                    p.add_(gparam, alpha=-lr)
                else:
                    # ZO-Adam / ZO-AdamW
                    grad_est = projected_grad * z

                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    state["step"] += 1
                    step_t = state["step"]

                    # coupled weight decay (Adam-style)
                    if weight_decay != 0.0 and not decoupled_wd:
                        grad_est.add_(p.data, alpha=weight_decay)

                    # update first and second moment running average
                    exp_avg.mul_(beta1).add_(grad_est, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_est, grad_est, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** step_t
                    bias_correction2 = 1 - beta2 ** step_t

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(adam_eps)
                    step_size = lr / bias_correction1

                    # decoupled weight decay (AdamW)
                    if weight_decay != 0.0 and decoupled_wd:
                        p.data.mul_(1 - lr * weight_decay)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return 0.5 * (loss_plus_val + loss_minus_val)

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
        - 'sgd'         : ZO-SGD
        - 'adam'        : ZO-Adam (coupled weight decay)
        - 'adamw'       : ZO-AdamW (decoupled weight decay)
        - 'adam_adapt'  : ZO-Adam + bold-driver style LR adaptation on loss
        - 'adam_adapt2' : ZO-Adam + gradient-statistics-based LR adaptation
        - 'adam_adapt3' : ZO-Adam + combined loss+gradient LR adaptation
                          (geometric mix of the two above)

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
        variant: str = "sgd",   # 'sgd', 'adam', 'adamw', 'adam_adapt', 'adam_adapt2', 'adam_adapt3'
        betas=(0.9, 0.999),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        projected_grad_clip: float | None = 5.0,
        # adaptive-LR controls (used by adapt/adapt2/adapt3)
        lr_min_factor: float = 0.1,
        lr_max_factor: float = 5.0,
        lr_inc_factor: float = 0.02,
        lr_dec_factor: float = 0.5,
        adapt_warmup: int = 10,
        adapt_every: int = 5,
        adapt_worsen_tol: float = 1e-3,
        # extra controls for the gradient-statistics variant ('adam_adapt2'/'adam_adapt3')
        adapt2_beta: float = 0.95,
        adapt2_alpha: float = 0.1,
        adapt2_eps: float = 1e-8,
        # mix weight for loss vs grad in adam_adapt3 (0 = only loss, 1 = only grad)
        adapt3_mix_grad: float = 0.5,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if variant not in ("sgd", "adam", "adamw", "adam_adapt", "adam_adapt2", "adam_adapt3"):
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
            # shared adaptive-LR defaults
            lr_min_factor=lr_min_factor,
            lr_max_factor=lr_max_factor,
            lr_inc_factor=lr_inc_factor,
            lr_dec_factor=lr_dec_factor,
            adapt_warmup=adapt_warmup,
            adapt_every=adapt_every,
            adapt_worsen_tol=adapt_worsen_tol,
            # gradient-statistics variant
            adapt2_beta=adapt2_beta,
            adapt2_alpha=adapt2_alpha,
            adapt2_eps=adapt2_eps,
            # combined variant
            adapt3_mix_grad=adapt3_mix_grad,
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

    # ---------------- adaptive LR helper ----------------

    def _update_adaptive_lr(self, avg_loss: float, projected_grad: float) -> None:
        """
        Update per-group learning rates for adaptive variants.

        - 'adam_adapt'  : bold-driver style, using the ZO loss.
        - 'adam_adapt2' : gradient-statistics style, using EMA of g_proj^2.
        - 'adam_adapt3' : combines both in a geometric/log-space mix.
        """
        if not math.isfinite(avg_loss):
            # If the loss is already non-finite, don't try to adapt LR from it.
            return

        for group in self.param_groups:
            variant = group.get("variant", None)
            if variant not in ("adam_adapt", "adam_adapt2", "adam_adapt3"):
                continue

            lr_min = group["lr_min_factor"]
            lr_max = group["lr_max_factor"]
            warmup = group["adapt_warmup"]
            adapt_every = group["adapt_every"]

            if variant == "adam_adapt":
                # -------- loss-based bold-driver adaptation --------
                adapt_state = group.setdefault("_adapt_state", {})
                base_lr = adapt_state.get("base_lr", group["lr"])
                adapt_state["base_lr"] = base_lr

                step = adapt_state.get("step", 0) + 1
                adapt_state["step"] = step

                best_loss = adapt_state.get("best_loss", None)
                lr_scale = adapt_state.get("lr_scale", 1.0)

                lr_inc = group["lr_inc_factor"]
                lr_dec = group["lr_dec_factor"]
                worsen_tol = group["adapt_worsen_tol"]

                # During warmup or non-adaptation steps, keep LR as-is.
                if step <= warmup or (step % adapt_every) != 0:
                    group["lr"] = base_lr * lr_scale
                    continue

                if best_loss is None or avg_loss < best_loss - 1e-12:
                    # Improvement: update best and gently increase lr.
                    best_loss = avg_loss
                    lr_scale = min(lr_scale * (1.0 + lr_inc), lr_max)
                elif avg_loss > best_loss * (1.0 + worsen_tol):
                    # Clear worsening: cut lr more aggressively.
                    lr_scale = max(lr_scale * lr_dec, lr_min)

                adapt_state["best_loss"] = best_loss
                adapt_state["lr_scale"] = lr_scale
                group["lr"] = base_lr * lr_scale

            elif variant == "adam_adapt2":
                # -------- gradient-statistics based adaptation --------
                adapt_state = group.setdefault("_adapt2_state", {})
                base_lr = adapt_state.get("base_lr", group["lr"])
                adapt_state["base_lr"] = base_lr

                step = adapt_state.get("step", 0) + 1
                adapt_state["step"] = step

                lr_scale = adapt_state.get("lr_scale", 1.0)
                gp2_ema = adapt_state.get("gp2_ema", projected_grad * projected_grad)

                beta = group["adapt2_beta"]
                alpha = group["adapt2_alpha"]
                eps_g = group["adapt2_eps"]

                # track EMA of projected_grad^2
                gp2_ema = beta * gp2_ema + (1.0 - beta) * (projected_grad * projected_grad)
                adapt_state["gp2_ema"] = gp2_ema

                # During warmup / non-adapt steps, just keep LR as-is
                if step <= warmup or (step % adapt_every) != 0:
                    group["lr"] = base_lr * lr_scale
                    adapt_state["lr_scale"] = lr_scale
                    continue

                gp_rms = math.sqrt(gp2_ema) + eps_g

                # Establish a reference gradient scale from early training
                ref_gp_rms = adapt_state.get("ref_gp_rms", None)
                if ref_gp_rms is None:
                    ref_gp_rms = gp_rms
                    adapt_state["ref_gp_rms"] = ref_gp_rms
                    group["lr"] = base_lr * lr_scale
                    adapt_state["lr_scale"] = lr_scale
                    continue

                # Target LR scale inversely proportional to gradient RMS
                lr_scale_target = ref_gp_rms / gp_rms

                # Clamp, then smooth to avoid jitter
                lr_scale_target = max(lr_min, min(lr_max, lr_scale_target))
                lr_scale = (1.0 - alpha) * lr_scale + alpha * lr_scale_target

                adapt_state["lr_scale"] = lr_scale
                group["lr"] = base_lr * lr_scale

            elif variant == "adam_adapt3":
                # -------- combined loss + gradient adaptation --------
                adapt_state = group.setdefault("_adapt3_state", {})
                base_lr = adapt_state.get("base_lr", group["lr"])
                adapt_state["base_lr"] = base_lr

                step = adapt_state.get("step", 0) + 1
                adapt_state["step"] = step

                lr_scale = adapt_state.get("lr_scale", 1.0)

                # loss-based pieces
                best_loss = adapt_state.get("best_loss", None)
                lr_inc = group["lr_inc_factor"]
                lr_dec = group["lr_dec_factor"]
                worsen_tol = group["adapt_worsen_tol"]

                # grad-based pieces
                gp2_ema = adapt_state.get("gp2_ema", projected_grad * projected_grad)
                beta = group["adapt2_beta"]
                alpha = group["adapt2_alpha"]   # reuse same smoothing factor
                eps_g = group["adapt2_eps"]
                mix_grad = group["adapt3_mix_grad"]

                # update EMA of grad^2
                gp2_ema = beta * gp2_ema + (1.0 - beta) * (projected_grad * projected_grad)
                adapt_state["gp2_ema"] = gp2_ema
                gp_rms = math.sqrt(gp2_ema) + eps_g

                # During warmup / non-adapt steps, just keep LR
                if step <= warmup or (step % adapt_every) != 0:
                    group["lr"] = base_lr * lr_scale
                    adapt_state["lr_scale"] = lr_scale
                    continue

                # ---- 1) loss-based suggestion s_loss ----
                s_loss = lr_scale
                if best_loss is None:
                    best_loss = avg_loss
                else:
                    if avg_loss < best_loss - 1e-12:
                        best_loss = avg_loss
                        s_loss = min(lr_scale * (1.0 + lr_inc), lr_max)
                    elif avg_loss > best_loss * (1.0 + worsen_tol):
                        s_loss = max(lr_scale * lr_dec, lr_min)
                    # else: s_loss = lr_scale (no change)

                adapt_state["best_loss"] = best_loss

                # ---- 2) grad-based suggestion s_grad ----
                ref_gp_rms = adapt_state.get("ref_gp_rms", None)
                if ref_gp_rms is None:
                    ref_gp_rms = gp_rms
                    adapt_state["ref_gp_rms"] = ref_gp_rms
                    s_grad = lr_scale  # no change yet
                else:
                    s_grad_target = ref_gp_rms / gp_rms
                    s_grad_target = max(lr_min, min(lr_max, s_grad_target))
                    s_grad = s_grad_target

                # ---- 3) mix in log-space between s_loss and s_grad ----
                # clamp to avoid log(0)
                s_loss_clamped = max(lr_min, min(lr_max, s_loss))
                s_grad_clamped = max(lr_min, min(lr_max, s_grad))

                log_s_loss = math.log(s_loss_clamped)
                log_s_grad = math.log(s_grad_clamped)
                # mix_grad = 0 → only loss; 1 → only grad
                log_target = (1.0 - mix_grad) * log_s_loss + mix_grad * log_s_grad
                lr_scale_target = math.exp(log_target)

                # ---- 4) smooth towards target, clamp ----
                lr_scale_target = max(lr_min, min(lr_max, lr_scale_target))
                lr_scale = (1.0 - alpha) * lr_scale + alpha * lr_scale_target

                adapt_state["lr_scale"] = lr_scale
                group["lr"] = base_lr * lr_scale

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

        # ---------------- non-finite guard on losses ----------------
        if (not math.isfinite(loss_plus_val)) or (not math.isfinite(loss_minus_val)):
            # restore parameters to original θ and shrink LR a bit
            print(
                f"[MeZO] Non-finite ZO loss detected: "
                f"loss_plus={loss_plus_val}, loss_minus={loss_minus_val}. "
                f"Restoring parameters and reducing lr.",
                flush=True,
            )
            for group in self.param_groups:
                epsilon = group["epsilon"]
                params = group["params"]
                if not params:
                    continue
                for p in params:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]
                    z = state.get("z", None)
                    if z is None:
                        continue
                    # currently θ = θ - εz, so restore by +εz
                    p.add_(epsilon * z)
                # gentle LR backoff
                group["lr"] *= 0.5
            # return a big but finite loss so logs don't show NaNs
            return float("inf")

        # scalar projected gradient
        projected_grad = (loss_plus_val - loss_minus_val) / (2.0 * eps)

        # clip projected gradient if requested
        if proj_clip is not None:
            if projected_grad > proj_clip:
                projected_grad = proj_clip
            elif projected_grad < -proj_clip:
                projected_grad = -proj_clip

        # if something still went wrong, zero it out
        if not math.isfinite(projected_grad):
            projected_grad = 0.0

        # Average loss (used for adaptive LR variants)
        avg_loss = 0.5 * (loss_plus_val + loss_minus_val)

        # Update any adaptive learning rates
        self._update_adaptive_lr(avg_loss, projected_grad)

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
            use_adam = group["variant"] in (
                "adam",
                "adamw",
                "adam_adapt",
                "adam_adapt2",
                "adam_adapt3",
            )

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
                elif use_adam:
                    # ZO-Adam / ZO-AdamW / ZO-Adam with adaptive LR
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
                else:
                    raise RuntimeError(f"Unsupported MeZO variant in step(): {group['variant']}")

        return avg_loss

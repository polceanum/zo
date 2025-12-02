import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm


# ================================================================
# Objective surfaces (with safe clipping)
# ================================================================

def rosenbrock_fn(x, y):
    x_safe = np.clip(x, -20.0, 20.0)
    y_safe = np.clip(y, -20.0, 20.0)
    return 100.0 * (y_safe - x_safe**2) ** 2 + (1 - x_safe) ** 2


def quadratic_fn(x, y, cond=10.0):
    return 0.5 * (1.0 * x**2 + cond * y**2)


def rastrigin_fn(x, y, A=10.0):
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))


def ackley_fn(x, y, a=20.0, b=0.2, c=2 * np.pi):
    s1 = (x**2 + y**2) / 2.0
    s2 = (np.cos(c * x) + np.cos(c * y)) / 2.0
    return -a * np.exp(-b * np.sqrt(s1 + 1e-12)) - np.exp(s2) + a + np.e


def l1_fn(x, y):
    return np.abs(x) + np.abs(y)


def build_surface(problem, xlim, ylim, resolution=200, cond=10.0):
    """
    Build a 2D surface for visualization. For noisy_quadratic we just
    show the underlying quadratic bowl (noise is per-eval, not in the
    analytic surface).
    """
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    if problem in ("quadratic", "noisy_quadratic"):
        Z = quadratic_fn(X, Y, cond=cond)
    elif problem == "sphere":
        Z = 0.5 * (X**2 + Y**2)
    elif problem == "rosenbrock":
        Z = rosenbrock_fn(X, Y)
    elif problem == "rastrigin":
        Z = rastrigin_fn(X, Y)
    elif problem == "ackley":
        Z = ackley_fn(X, Y)
    elif problem == "l1":
        Z = l1_fn(X, Y)
    else:
        raise ValueError(f"Unknown problem: {problem}")

    return X, Y, Z


# ================================================================
# Trajectory loading / cleaning
# ================================================================

def load_raw_runs(traj_dir, problem, optimizers=None):
    metas = sorted(glob.glob(os.path.join(traj_dir, f"{problem}_*_meta.json")))
    runs = []
    for meta_path in metas:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        opt = meta["optimizer"]
        if optimizers is not None and opt not in optimizers:
            continue
        traj = np.load(meta["traj_path"])
        # need at least 2D parameter vectors to plot
        if traj.shape[1] < 2:
            continue
        runs.append((opt, meta, traj))
    return runs


def clean_traj(traj: np.ndarray):
    """
    Remove tail after first NaN/Inf in the first two coordinates.
    Returns cleaned traj, or None if no finite points at all.
    """
    pts = traj[:, :2]
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        return None
    last = np.where(finite)[0].max()
    return traj[: last + 1]


# ================================================================
# Animation
# ================================================================

def make_animation(
    problem,
    traj_dir,
    output,
    optimizers=None,
    radius=2.0,
    max_center_abs=50.0,
):
    raw_runs = load_raw_runs(traj_dir, problem, optimizers)
    if not raw_runs:
        raise RuntimeError("No trajectories found.")

    runs = []
    finals = []

    for opt, meta, traj in raw_runs:
        c = clean_traj(traj)
        if c is None:
            print(f"[WARN] Skipping {opt}: non-finite trajectory")
            continue
        runs.append((opt, meta, c))
        finals.append(c[-1, :2])

    if not runs:
        raise RuntimeError("All trajectories non-finite.")

    finals = np.stack(finals)
    finite_mask = np.isfinite(finals).all(axis=1)
    finals = finals[finite_mask]
    if finals.size == 0:
        raise RuntimeError("Convergence points are all NaN/Inf.")

    # pick "reasonable" convergence region to center the view
    abs_ok = (np.abs(finals[:, 0]) <= max_center_abs) & (np.abs(finals[:, 1]) <= max_center_abs)
    center_pts = finals[abs_ok] if abs_ok.any() else np.clip(finals, -max_center_abs, max_center_abs)

    cx = 0.5 * (center_pts[:, 0].min() + center_pts[:, 0].max())
    cy = 0.5 * (center_pts[:, 1].min() + center_pts[:, 1].max())

    r = max(radius, 1e-3)
    xlim = (cx - r, cx + r)
    ylim = (cy - r, cy + r)

    if not np.all(np.isfinite(xlim)) or xlim[0] == xlim[1]:
        xlim = (cx - 1, cx + 1)
    if not np.all(np.isfinite(ylim)) or ylim[0] == ylim[1]:
        ylim = (cy - 1, cy + 1)

    # Background contour
    X, Y, Z = build_surface(problem, xlim, ylim)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(X, Y, Z, levels=30)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{problem} – optimization trajectories")

    # Lines & points, consistent color per optimizer
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lines = []
    points = []
    for idx, (opt, meta, traj) in enumerate(runs):
        c = color_cycle[idx % len(color_cycle)]
        (line,) = ax.plot([], [], lw=2, color=c, label=opt)
        (pt,) = ax.plot([], [], "o", color=c, ms=5)
        lines.append(line)
        points.append(pt)

    fig.subplots_adjust(right=0.75)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    max_steps = max(t.shape[0] for _, _, t in runs)

    def init():
        for l, p in zip(lines, points):
            l.set_data([], [])
            p.set_data([], [])
        return lines + points

    def animate_frame(t):
        for (opt, meta, traj), line, pt in zip(runs, lines, points):
            T = traj.shape[0]
            i = min(t, T - 1)
            xs = traj[: i + 1, 0]
            ys = traj[: i + 1, 1]
            line.set_data(xs, ys)
            pt.set_data([xs[-1]], [ys[-1]])
        return lines + points

    anim = animation.FuncAnimation(
        fig, animate_frame, init_func=init,
        frames=max_steps, interval=40, blit=True
    )

    # Progress bar during saving
    pbar = tqdm(total=max_steps, desc="Rendering video")

    def progress_callback(frame_number, total_frames):
        pbar.update(1)
        if frame_number + 1 == total_frames:
            pbar.close()

    writer = "pillow" if output.lower().endswith(".gif") else "ffmpeg"

    anim.save(output, writer=writer, progress_callback=progress_callback)
    print(f"Saved animation → {output}")


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--traj-dir", default="toy_trajectories")
    parser.add_argument("--output", default="anim.mp4")
    parser.add_argument("--optimizers", nargs="*", default=None)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--max-center-abs", type=float, default=50.0)
    args = parser.parse_args()

    make_animation(
        problem=args.problem,
        traj_dir=args.traj_dir,
        output=args.output,
        optimizers=args.optimizers,
        radius=args.radius,
        max_center_abs=args.max_center_abs,
    )


if __name__ == "__main__":
    main()

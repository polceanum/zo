#!/usr/bin/env python3
"""
results_diff.py

CI-friendly diff of two results.json files.

- Matches runs by (problem, optimizer, seed, full config)
- Compares numeric metrics with abs/rel tolerances
- Reports largest deltas (test loss, train loss, accuracy, etc.)
- Exits non-zero on regression (or missing runs, by default)

Expected results.json schema:
{
  "schema_version": 1,
  "runs": {
    "<run_id>": {
      "config": {...},
      "metrics": {...}
    },
    ...
  }
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------

def _is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return False


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_results(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _fmt_float(x: Any, width: int = 12, prec: int = 6) -> str:
    if x is None:
        return f"{'':>{width}}"
    if not _is_number(x):
        return f"{str(x):>{width}}"
    v = float(x)
    # Use scientific for very small/large values, fixed otherwise.
    if v == 0.0:
        s = "0"
    elif abs(v) < 1e-4 or abs(v) >= 1e5:
        s = f"{v:.{prec}e}"
    else:
        s = f"{v:.{prec}f}"
    return f"{s:>{width}}"


def _short_run_id(run_id: str) -> str:
    return run_id[:10] if isinstance(run_id, str) and len(run_id) >= 10 else str(run_id)


def _metric_direction(metric_name: str) -> str:
    """
    Returns:
      - "lower" => lower is better (loss-like)
      - "higher" => higher is better (accuracy-like)
      - "lower_or_equal" => lower is better (time-like; allow equal)
      - "unknown" => do not classify as regression unless explicitly asked
    """
    m = metric_name.lower()
    if "loss" in m:
        return "lower"
    if "acc" in m or "accuracy" in m:
        return "higher"
    if "time" in m or "sec" in m:
        return "lower_or_equal"
    return "unknown"


@dataclass(frozen=True)
class RunKey:
    kind: str
    problem: str
    optimizer: str
    seed: int
    config_json: str  # full config, stabilized


@dataclass
class RunEntry:
    run_id: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    key: RunKey

    @property
    def label(self) -> str:
        # Stable, human-readable identifier.
        kind = self.key.kind
        problem = self.key.problem
        opt = self.key.optimizer
        seed = self.key.seed
        rid = _short_run_id(self.run_id)
        return f"{kind}:{problem}:{opt}:seed{seed} ({rid})"


def _build_run_key(cfg: Dict[str, Any]) -> RunKey:
    kind = str(cfg.get("kind", ""))
    problem = str(cfg.get("problem", ""))
    optimizer = str(cfg.get("optimizer", ""))
    seed_raw = cfg.get("seed", 0)
    try:
        seed = int(seed_raw)
    except Exception:
        seed = 0
    return RunKey(
        kind=kind,
        problem=problem,
        optimizer=optimizer,
        seed=seed,
        config_json=_stable_json(cfg),
    )


def _index_runs(db: Dict[str, Any]) -> Dict[RunKey, RunEntry]:
    runs = db.get("runs", {})
    if not isinstance(runs, dict):
        raise ValueError("results.json: expected top-level 'runs' to be a dict")

    out: Dict[RunKey, RunEntry] = {}
    for run_id, entry in runs.items():
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config", {})
        met = entry.get("metrics", {})
        if not isinstance(cfg, dict) or not isinstance(met, dict):
            continue

        key = _build_run_key(cfg)
        if key in out:
            # Duplicate key: keep the first, but warn via stderr.
            print(
                f"WARNING: duplicate run key detected for {key.kind}:{key.problem}:{key.optimizer}:seed{key.seed}. "
                f"Keeping first (run_id={_short_run_id(out[key].run_id)}) and ignoring run_id={_short_run_id(str(run_id))}.",
                file=sys.stderr,
            )
            continue

        out[key] = RunEntry(
            run_id=str(run_id),
            config=cfg,
            metrics=met,
            key=key,
        )
    return out


def _parse_metric_tolerances(kv_list: Optional[List[str]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Parse:
      --metric-abs-tol name=val
      --metric-rel-tol name=val
    Returns (abs_tols, rel_tols)
    """
    abs_tols: Dict[str, float] = {}
    rel_tols: Dict[str, float] = {}

    if not kv_list:
        return abs_tols, rel_tols

    for kv in kv_list:
        if "=" not in kv:
            raise ValueError(f"Invalid metric tolerance '{kv}'. Expected format name=value")
        name, val = kv.split("=", 1)
        name = name.strip()
        val = val.strip()
        if not name:
            raise ValueError(f"Invalid metric tolerance '{kv}': empty metric name")
        try:
            fval = float(val)
        except Exception as e:
            raise ValueError(f"Invalid metric tolerance '{kv}': value is not float") from e

        # The caller decides whether it belongs to abs or rel by passing separate lists.
        # We just return them in two dicts (caller assigns).
        abs_tols[name] = fval  # temporary; caller can re-map
    return abs_tols, rel_tols


def _apply_metric_tol_overrides(
    base_abs_tol: float,
    base_rel_tol: float,
    metric_name: str,
    metric_abs_overrides: Dict[str, float],
    metric_rel_overrides: Dict[str, float],
) -> Tuple[float, float]:
    abs_tol = base_abs_tol
    rel_tol = base_rel_tol
    if metric_name in metric_abs_overrides:
        abs_tol = metric_abs_overrides[metric_name]
    if metric_name in metric_rel_overrides:
        rel_tol = metric_rel_overrides[metric_name]
    return abs_tol, rel_tol


def _delta_exceeds_tol(
    base: float,
    cand: float,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    """
    True if |cand - base| > abs_tol + rel_tol * max(|base|, 1).
    """
    diff = abs(cand - base)
    denom = max(abs(base), 1.0)
    allowed = abs_tol + rel_tol * denom
    return diff > allowed


@dataclass
class MetricDiff:
    metric: str
    base: float
    cand: float
    delta: float  # cand - base
    abs_delta: float
    rel_delta: float  # abs_delta / max(|base|, 1)
    abs_tol: float
    rel_tol: float
    direction: str
    is_regression: bool


@dataclass
class RunDiff:
    key: RunKey
    base: RunEntry
    cand: RunEntry
    metric_diffs: List[MetricDiff]


def _compute_metric_diffs(
    base_run: RunEntry,
    cand_run: RunEntry,
    abs_tol: float,
    rel_tol: float,
    metric_abs_overrides: Dict[str, float],
    metric_rel_overrides: Dict[str, float],
    consider_unknown_metrics: bool,
) -> List[MetricDiff]:
    base_m = base_run.metrics
    cand_m = cand_run.metrics

    common_metrics = sorted(set(base_m.keys()) & set(cand_m.keys()))
    diffs: List[MetricDiff] = []

    for k in common_metrics:
        bv = base_m.get(k, None)
        cv = cand_m.get(k, None)
        if not (_is_number(bv) and _is_number(cv)):
            continue

        b = float(bv)
        c = float(cv)

        mt_abs, mt_rel = _apply_metric_tol_overrides(
            abs_tol, rel_tol, k, metric_abs_overrides, metric_rel_overrides
        )

        delta = c - b
        abs_delta = abs(delta)
        rel_delta = abs_delta / max(abs(b), 1.0)
        direction = _metric_direction(k)

        # Determine regression by direction + tolerance.
        exceeds = _delta_exceeds_tol(b, c, mt_abs, mt_rel)

        if direction == "lower":
            is_reg = exceeds and (c > b)
        elif direction == "higher":
            is_reg = exceeds and (c < b)
        elif direction == "lower_or_equal":
            is_reg = exceeds and (c > b)
        else:
            is_reg = bool(consider_unknown_metrics and exceeds)

        diffs.append(
            MetricDiff(
                metric=k,
                base=b,
                cand=c,
                delta=delta,
                abs_delta=abs_delta,
                rel_delta=rel_delta,
                abs_tol=mt_abs,
                rel_tol=mt_rel,
                direction=direction,
                is_regression=is_reg,
            )
        )

    return diffs


def _pick_metric_value(run: RunEntry, candidates: Iterable[str]) -> Optional[float]:
    for k in candidates:
        v = run.metrics.get(k, None)
        if _is_number(v):
            return float(v)
    return None


def _get_primary_metrics(kind: str) -> Dict[str, List[str]]:
    """
    Define "useful" primary metrics per kind for sorting / summaries.

    - ML: use final_test_loss (or final_eval_loss) and accuracy.
    - Toy: use final_loss and best_loss.
    """
    if kind == "ml":
        return {
            "test_loss": ["final_test_loss", "final_eval_loss", "test_loss", "eval_loss"],
            "train_loss": ["train_loss", "final_train_loss"],
            "accuracy": ["final_test_acc", "final_eval_acc", "test_acc", "eval_acc", "accuracy"],
            "best_test_loss": ["best_test_loss", "best_eval_loss"],
            "best_accuracy": ["best_test_acc", "best_eval_acc"],
        }
    return {
        "train_loss": ["final_loss"],
        "best_train_loss": ["best_loss"],
        "test_loss": ["final_test_loss", "test_loss"],  # usually absent for toy
        "accuracy": ["final_test_acc", "accuracy"],     # usually absent for toy
    }


def _print_top_deltas(
    title: str,
    run_diffs: List[RunDiff],
    metric_name_candidates: List[str],
    top_n: int,
) -> None:
    # Compute deltas for the first available metric per run.
    rows = []
    for rd in run_diffs:
        b = _pick_metric_value(rd.base, metric_name_candidates)
        c = _pick_metric_value(rd.cand, metric_name_candidates)
        if b is None or c is None:
            continue
        rows.append((abs(c - b), c - b, b, c, rd.base.label))

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:top_n]

    if not rows:
        return

    print("\n" + title)
    print("-" * len(title))
    print(f"{'absΔ':>12} {'Δ':>12} {'base':>12} {'cand':>12}  run")
    print("-" * (12 + 1 + 12 + 1 + 12 + 1 + 12 + 2 + 3 + 60))
    for abs_d, d, b, c, label in rows:
        print(f"{_fmt_float(abs_d)} {_fmt_float(d)} {_fmt_float(b)} {_fmt_float(c)}  {label}")


def _print_regressions(regressions: List[Tuple[str, MetricDiff, str]], top_n: int) -> None:
    if not regressions:
        return
    regressions.sort(key=lambda t: (t[1].abs_delta, t[1].rel_delta), reverse=True)
    regressions = regressions[:top_n]

    print("\nREGRESSIONS (worst first)")
    print("------------------------")
    print(f"{'absΔ':>12} {'relΔ':>10} {'metric':<24} {'base':>12} {'cand':>12}  run")
    print("-" * (12 + 1 + 10 + 1 + 24 + 1 + 12 + 1 + 12 + 2 + 3 + 60))
    for run_label, md, _key_str in regressions:
        print(
            f"{_fmt_float(md.abs_delta)} "
            f"{md.rel_delta:>10.6f} "
            f"{md.metric:<24} "
            f"{_fmt_float(md.base)} "
            f"{_fmt_float(md.cand)}  "
            f"{run_label}"
        )


def _key_str(k: RunKey) -> str:
    return f"{k.kind}:{k.problem}:{k.optimizer}:seed{k.seed}:{k.config_json}"


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff two results.json files (CI-ready, non-zero exit on regressions)."
    )
    parser.add_argument("baseline", type=str, help="Baseline results.json (expected/golden)")
    parser.add_argument("candidate", type=str, help="Candidate results.json (new run to compare)")

    parser.add_argument("--abs-tol", type=float, default=1e-8, help="Default absolute tolerance")
    parser.add_argument("--rel-tol", type=float, default=1e-6, help="Default relative tolerance")
    parser.add_argument(
        "--metric-abs-tol",
        type=str,
        nargs="*",
        default=[],
        help="Per-metric abs tol overrides: name=value (exact metric key match)",
    )
    parser.add_argument(
        "--metric-rel-tol",
        type=str,
        nargs="*",
        default=[],
        help="Per-metric rel tol overrides: name=value (exact metric key match)",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many top deltas/regressions to print",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        default=True,
        help="Fail if a baseline run is missing in candidate (default: True)",
    )
    parser.add_argument(
        "--no-fail-on-missing",
        action="store_false",
        dest="fail_on_missing",
        help="Do not fail if baseline runs are missing in candidate",
    )
    parser.add_argument(
        "--fail-on-added",
        action="store_true",
        default=False,
        help="Fail if candidate contains runs not present in baseline (default: False)",
    )
    parser.add_argument(
        "--consider-unknown-metrics",
        action="store_true",
        default=False,
        help="Treat unknown-direction metrics as regressions if tolerance exceeded (default: False)",
    )

    args = parser.parse_args()

    base_db = _load_results(args.baseline)
    cand_db = _load_results(args.candidate)

    base_runs = _index_runs(base_db)
    cand_runs = _index_runs(cand_db)

    # Parse per-metric overrides
    metric_abs_overrides: Dict[str, float] = {}
    metric_rel_overrides: Dict[str, float] = {}

    if args.metric_abs_tol:
        for kv in args.metric_abs_tol:
            if "=" not in kv:
                raise ValueError(f"Invalid --metric-abs-tol '{kv}'. Expected name=value.")
            name, val = kv.split("=", 1)
            metric_abs_overrides[name.strip()] = float(val.strip())

    if args.metric_rel_tol:
        for kv in args.metric_rel_tol:
            if "=" not in kv:
                raise ValueError(f"Invalid --metric-rel-tol '{kv}'. Expected name=value.")
            name, val = kv.split("=", 1)
            metric_rel_overrides[name.strip()] = float(val.strip())

    base_keys = set(base_runs.keys())
    cand_keys = set(cand_runs.keys())

    missing = sorted(base_keys - cand_keys, key=_key_str)
    added = sorted(cand_keys - base_keys, key=_key_str)
    common = sorted(base_keys & cand_keys, key=_key_str)

    print("RESULTS DIFF")
    print("------------")
    print(f"Baseline : {args.baseline}")
    print(f"Candidate: {args.candidate}")
    print(f"Runs: baseline={len(base_keys)} candidate={len(cand_keys)} common={len(common)}")
    print(f"Missing baseline runs in candidate: {len(missing)}")
    print(f"Added candidate runs vs baseline:   {len(added)}")
    print(f"Tolerances: abs={args.abs_tol} rel={args.rel_tol}")
    if metric_abs_overrides:
        print(f"Metric abs overrides: {metric_abs_overrides}")
    if metric_rel_overrides:
        print(f"Metric rel overrides: {metric_rel_overrides}")

    # Show missing / added (limited)
    if missing:
        print("\nMISSING (baseline present, candidate absent)")
        print("------------------------------------------")
        for k in missing[: max(1, args.top)]:
            print(f"- {base_runs[k].label}")
        if len(missing) > args.top:
            print(f"... and {len(missing) - args.top} more")

    if added:
        print("\nADDED (candidate present, baseline absent)")
        print("----------------------------------------")
        for k in added[: max(1, args.top)]:
            print(f"+ {cand_runs[k].label}")
        if len(added) > args.top:
            print(f"... and {len(added) - args.top} more")

    # Compute diffs for common runs
    run_diffs: List[RunDiff] = []
    regressions: List[Tuple[str, MetricDiff, str]] = []

    for k in common:
        b = base_runs[k]
        c = cand_runs[k]
        mds = _compute_metric_diffs(
            base_run=b,
            cand_run=c,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            metric_abs_overrides=metric_abs_overrides,
            metric_rel_overrides=metric_rel_overrides,
            consider_unknown_metrics=args.consider_unknown_metrics,
        )
        run_diffs.append(RunDiff(key=k, base=b, cand=c, metric_diffs=mds))

        for md in mds:
            if md.is_regression:
                regressions.append((b.label, md, _key_str(k)))

    # Print top deltas by useful primary metrics
    if run_diffs:
        # Partition by kind for clearer summaries
        by_kind: Dict[str, List[RunDiff]] = {}
        for rd in run_diffs:
            by_kind.setdefault(rd.key.kind, []).append(rd)

        for kind, rds in sorted(by_kind.items(), key=lambda kv: kv[0]):
            prim = _get_primary_metrics(kind)

            _print_top_deltas(
                title=f"TOP |Δ| by TEST LOSS ({kind})",
                run_diffs=rds,
                metric_name_candidates=prim.get("test_loss", []),
                top_n=args.top,
            )
            _print_top_deltas(
                title=f"TOP |Δ| by TRAIN LOSS ({kind})",
                run_diffs=rds,
                metric_name_candidates=prim.get("train_loss", []),
                top_n=args.top,
            )
            _print_top_deltas(
                title=f"TOP |Δ| by ACCURACY ({kind})",
                run_diffs=rds,
                metric_name_candidates=prim.get("accuracy", []),
                top_n=args.top,
            )
            # Also include best_* if present
            _print_top_deltas(
                title=f"TOP |Δ| by BEST TEST LOSS ({kind})",
                run_diffs=rds,
                metric_name_candidates=prim.get("best_test_loss", []),
                top_n=args.top,
            )
            _print_top_deltas(
                title=f"TOP |Δ| by BEST ACCURACY ({kind})",
                run_diffs=rds,
                metric_name_candidates=prim.get("best_accuracy", []),
                top_n=args.top,
            )

    _print_regressions(regressions, top_n=args.top)

    # Decide CI status
    failed = False

    if regressions:
        failed = True
        print(f"\n❌ Regression(s) detected: {len(regressions)}")

    if args.fail_on_missing and missing:
        failed = True
        print(f"\n❌ Missing baseline runs in candidate: {len(missing)}")

    if args.fail_on_added and added:
        failed = True
        print(f"\n❌ Added runs in candidate (fail-on-added enabled): {len(added)}")

    if not failed:
        print("\n✅ No regressions detected within tolerances.")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Allow piping to tools like `head` without stack traces.
        raise SystemExit(1)

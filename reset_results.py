#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Any, List, Optional


def load_results(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"No results file at {path}")
        # Keep compatibility with both schema v1/v2
        return {"schema_version": 1, "runs": {}}
    with open(path, "r") as f:
        return json.load(f)


def save_results(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    print(f"Updated {path}")


def run_matches(cfg: Dict[str, Any], optimizers: Optional[List[str]], problem: Optional[str],
                kind: Optional[str], seeds: Optional[List[int]]) -> bool:
    """Return True if a run entry should be deleted based on filters."""

    if optimizers is not None and cfg.get("optimizer") not in optimizers:
        return False

    if problem is not None and cfg.get("problem") != problem:
        return False

    if kind is not None and cfg.get("kind") != kind:
        return False

    if seeds is not None and cfg.get("seed") not in seeds:
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely delete selected entries from the results DB JSON"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Path to results DB JSON (default: results.json)",
    )

    # Use '+' so the flag cannot be provided with zero values (prevents accidental 'match all')
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=None,
        help="Only remove runs for these optimizers (e.g. mezo_sgd mezo_adamu sgd adam)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Only remove runs for this problem/dataset (e.g. mnist, cifar10, quadratic)",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default=None,
        choices=["toy", "ml"],
        help="Only remove toy or ml entries",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only remove runs with these seeds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching entries but do not delete",
    )

    args = parser.parse_args()

    # Safe-by-default: refuse to delete everything if *no* filter is provided.
    if args.optimizers is None and args.problem is None and args.kind is None and args.seeds is None:
        print("\n🚫 No filters provided. This script is SAFE-BY-DEFAULT.")
        print("Running without filters would delete ALL runs — so nothing is done.\n")
        print("👉 Provide at least one of:")
        print("   --optimizers OPT1 OPT2 ...")
        print("   --problem mnist")
        print("   --kind toy|ml")
        print("   --seeds 1 2 3\n")
        print("Example:")
        print("   python reset_results.py --optimizers mezo_sgd --dry-run\n")
        return

    db = load_results(args.results)
    runs = db.get("runs", {})

    to_delete: List[str] = []
    for run_id, entry in runs.items():
        cfg = entry.get("config", {})
        if run_matches(cfg, args.optimizers, args.problem, args.kind, args.seeds):
            to_delete.append(run_id)

    if not to_delete:
        print("No matching entries found.")
        return

    print(f"Matched {len(to_delete)} entries:")
    for rid in to_delete:
        cfg = runs[rid].get("config", {})
        print(
            f" - {rid[:10]} | {cfg.get('kind')} | {cfg.get('problem')} | "
            f"{cfg.get('optimizer')} | seed={cfg.get('seed')}"
        )

    if args.dry_run:
        print("\nDry run: nothing deleted.")
        return

    for rid in to_delete:
        del runs[rid]

    db["runs"] = runs
    save_results(args.results, db)


if __name__ == "__main__":
    main()

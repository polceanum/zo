#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any


def load_results(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"No results file at {path}")
        return {"schema_version": 1, "runs": {}}
    with open(path, "r") as f:
        return json.load(f)


def save_results(path: str, data: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    print(f"Updated {path}")


def run_matches(cfg: Dict[str, Any], args: argparse.Namespace) -> bool:
    """
    Returns True if a run entry should be deleted based on filters.
    """

    # Filter by optimizer
    if args.optimizers and cfg.get("optimizer") not in args.optimizers:
        return False

    # Filter by problem (toy or ML)
    if args.problem and cfg.get("problem") != args.problem:
        return False

    # Filter by kind (toy or ml)
    if args.kind and cfg.get("kind") != args.kind:
        return False

    # Filter by seed
    if args.seeds and cfg.get("seed") not in args.seeds:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Safely reset (delete) selected entries from results.json"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Path to results.json file",
    )

    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="*",
        default=None,
        help="Optimizer names to clear (e.g. mezo_adam_smooth mezo_adam_tuned)",
    )

    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Specific problem/dataset to clear (e.g. mnist, quadratic)",
    )

    parser.add_argument(
        "--kind",
        type=str,
        default=None,
        choices=["toy", "ml"],
        help="Whether to clear only toy or ml entries",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Only remove runs with these seeds",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching entries but do not delete",
    )

    args = parser.parse_args()

    # ---------------- SAFE-BY-DEFAULT CHECK ----------------
    # If no filters at all, refuse to delete everything.
    if (
        args.optimizers is None
        and args.problem is None
        and args.kind is None
        and args.seeds is None
    ):
        print("\n🚫 No filters provided. This script is SAFE-BY-DEFAULT.")
        print("Running without filters would delete ALL runs — so nothing is done.\n")
        print("👉 Provide at least one of:")
        print("   --optimizers OPT1 OPT2 ...")
        print("   --problem mnist")
        print("   --kind toy|ml")
        print("   --seeds 1 2 3\n")
        print("Example:")
        print("   python reset_results.py --optimizers mezo_adam_tuned --dry-run\n")
        return
    # --------------------------------------------------------

    db = load_results(args.results)
    runs = db.get("runs", {})

    to_delete = []

    for run_id, entry in runs.items():
        cfg = entry.get("config", {})
        if run_matches(cfg, args):
            to_delete.append(run_id)

    if not to_delete:
        print("No matching entries found.")
        return

    print(f"Matched {len(to_delete)} entries:")
    for rid in to_delete:
        cfg = runs[rid]["config"]
        print(
            f" - {rid[:10]} | {cfg.get('kind')} | {cfg.get('problem')} | "
            f"{cfg.get('optimizer')} | seed={cfg.get('seed')}"
        )

    if args.dry_run:
        print("\nDry run: nothing deleted.")
        return

    # delete confirmed entries
    for rid in to_delete:
        del runs[rid]

    db["runs"] = runs
    save_results(args.results, db)


if __name__ == "__main__":
    main()

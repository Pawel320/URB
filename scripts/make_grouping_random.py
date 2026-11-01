"""Generate a random balanced 0/1 agent grouping CSV for SubAV baseline runs.

Writes 'agent_id,cluster' with labels 0/1 assigned uniformly at random
over the network's agents.csv (shuffled deterministically by --seed).
Useful as a baseline where clusters/classes are assigned at random instead
of being learned or hand-crafted (see make_grouping_roundrobin.py for the
deterministic parity variant used in smoke/validation runs).

Usage (from URB/scripts/):
    python scripts/make_grouping_random.py --net ingolstadt_custom2 \
        --out config/groupings/ingolstadt_custom2_random_seed42.csv \
        --seed 42 --ratio 0.5
"""

import argparse
import csv
import os
import random


def main():
    parser = argparse.ArgumentParser(
        description="Random balanced 0/1 grouping over a network's agents.csv."
    )
    parser.add_argument("--net", type=str, required=True, help="Network name under networks/.")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path (columns agent_id,cluster).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffle (default: 42).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Fraction of agents assigned to cluster 1 (default: 0.5).",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    agents_csv = os.path.join(repo_root, "networks", args.net, "agents.csv")
    if not os.path.exists(agents_csv):
        raise FileNotFoundError(f"Agents CSV not found: {agents_csv}")

    with open(agents_csv, newline="") as f:
        agent_ids = [row["id"] for row in csv.DictReader(f)]
    if not agent_ids:
        raise ValueError(f"No agents found in {agents_csv}")

    if not 0.0 < args.ratio < 1.0:
        raise ValueError(f"--ratio must be in (0, 1), got {args.ratio}")

    n1 = round(args.ratio * len(agent_ids))
    labels = [1] * n1 + [0] * (len(agent_ids) - n1)
    random.Random(args.seed).shuffle(labels)

    rows = [
        {"agent_id": aid, "cluster": lbl} for aid, lbl in zip(agent_ids, labels)
    ]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_id", "cluster"])
        writer.writeheader()
        writer.writerows(rows)

    n_cluster1 = sum(1 for row in rows if row["cluster"] == 1)
    print(
        f"Wrote {len(rows)} rows to {args.out} "
        f"(cluster 1: {n_cluster1}, cluster 0: {len(rows) - n_cluster1}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()

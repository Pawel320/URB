"""Generate a round-robin agent grouping CSV for SubAV smoke/validation runs.

Writes 'agent_id,cluster' with cluster = index % 2 over the network's agents.csv,
so whichever subset the environment turns into AVs, both clusters are populated.
For real experiments use your own externally computed grouping instead.

Usage (from URB root or scripts/):
    python scripts/make_grouping_roundrobin.py --net ingolstadt_custom2 \
        --out config/groupings/ingolstadt_custom2_roundrobin.csv
"""

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(description="Round-robin 0/1 grouping over a network's agents.csv.")
    parser.add_argument("--net", type=str, required=True, help="Network name under networks/.")
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path (columns agent_id,cluster).",
    )
    parser.add_argument("--parity", type=int, choices=[0, 1], default=0, help="Cluster label of the first agent.")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    agents_csv = os.path.join(repo_root, "networks", args.net, "agents.csv")
    if not os.path.exists(agents_csv):
        raise FileNotFoundError(f"Agents CSV not found: {agents_csv}")

    with open(agents_csv, newline="") as f:
        rows = [
            {"agent_id": row["id"], "cluster": (idx + args.parity) % 2}
            for idx, row in enumerate(csv.DictReader(f))
        ]
    if not rows:
        raise ValueError(f"No agents found in {agents_csv}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_id", "cluster"])
        writer.writeheader()
        writer.writerows(rows)

    n_cluster1 = sum(1 for row in rows if row["cluster"] == 1)
    print(f"Wrote {len(rows)} rows to {args.out} (cluster 1: {n_cluster1}, cluster 0: {len(rows) - n_cluster1})")


if __name__ == "__main__":
    main()

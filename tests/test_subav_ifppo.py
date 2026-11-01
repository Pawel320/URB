import csv
import os
import pytest
import shutil
import subprocess
import sys

from pathlib import Path

SCRIPTS_DIR = Path("scripts")
SUBAV_SCRIPT = SCRIPTS_DIR / "subav_ifppo.py"
AGENTS_CSV = Path("networks") / "saint_arnoult" / "agents.csv"
RESULTS_DIR = Path("results")
EXP_ID = "test_subav_ifppo"

SUMO_GATED = True


@pytest.fixture(scope="session", autouse=True)
def check_sumo_installed():
    sumo_executable = shutil.which("sumo")
    if sumo_executable is None:
        if SUMO_GATED:
            pytest.exit("[SUMO ERROR] SUMO is not installed or not in PATH.")
        else:
            pytest.skip("[SUMO SKIP] SUMO is not installed or not in PATH.")
    else:
        try:
            result = subprocess.run(
                ["sumo", "--version"], capture_output=True, text=True, check=True
            )
            print(f"[DEBUG] SUMO version: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            pytest.exit(f"[SUMO ERROR] Failed to get SUMO version: {e.stderr}")


def write_round_robin_grouping_csv(dest_path: Path) -> None:
    """Label every saint_arnoult agent alternately 0/1 so both clusters are
    populated whichever subset the environment turns into AVs."""

    with open(AGENTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            {"agent_id": row["id"], "cluster": idx % 2}
            for idx, row in enumerate(reader)
        ]

    assert rows, f"No agents found in {AGENTS_CSV}"
    with open(dest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent_id", "cluster"])
        writer.writeheader()
        writer.writerows(rows)


def test_subav_ifppo_execution(tmp_path):
    grouping_file = tmp_path / "grouping.csv"
    write_round_robin_grouping_csv(grouping_file)

    exp_results_dir = RESULTS_DIR / EXP_ID
    if exp_results_dir.exists():
        shutil.rmtree(exp_results_dir)

    try:
        result = subprocess.run(
            [sys.executable, SUBAV_SCRIPT.name,
             "--id", EXP_ID,
             "--alg-conf", "test",
             "--env-conf", "clusters-eta-mask",
             "--task-conf", "test",
             "--net", "saint_arnoult",
             "--env-seed", "42",
             "--grouping-file", str(grouping_file)],
            capture_output=True, text=True, check=True, cwd=SCRIPTS_DIR
        )
        print(f"[DEBUG] Successfully executed {SUBAV_SCRIPT}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] Script {SUBAV_SCRIPT} failed to execute: {e.stderr}")

    agent_clusters = exp_results_dir / "agent_clusters.csv"
    assert agent_clusters.exists(), f"Missing grouping copy at {agent_clusters}"

    cluster_metrics = exp_results_dir / "metrics" / "ClusterMetrics.csv"
    assert cluster_metrics.exists(), f"Missing cluster metrics at {cluster_metrics}"

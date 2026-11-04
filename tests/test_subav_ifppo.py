import csv
import json
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
SPLIT_EXP_ID = "test_subav_ifppo_obs_split"

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


def run_subav_experiment(exp_id: str, env_conf: str, tmp_path: Path) -> Path:
    """Run subav_ifppo.py end to end and return its results directory."""

    grouping_file = tmp_path / f"grouping_{exp_id}.csv"
    write_round_robin_grouping_csv(grouping_file)

    exp_results_dir = RESULTS_DIR / exp_id
    if exp_results_dir.exists():
        shutil.rmtree(exp_results_dir)

    try:
        subprocess.run(
            [sys.executable, SUBAV_SCRIPT.name,
             "--id", exp_id,
             "--alg-conf", "test",
             "--env-conf", env_conf,
             "--task-conf", "test",
             "--net", "saint_arnoult",
             "--env-seed", "42",
             "--grouping-file", str(grouping_file)],
            capture_output=True, text=True, check=True, cwd=SCRIPTS_DIR
        )
        print(f"[DEBUG] Successfully executed {SUBAV_SCRIPT} with --env-conf {env_conf}")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"[FAIL] Script {SUBAV_SCRIPT} failed to execute: {e.stderr}")

    return exp_results_dir


def test_subav_ifppo_execution(tmp_path):
    exp_results_dir = run_subav_experiment(EXP_ID, "clusters-eta-mask", tmp_path)

    agent_clusters = exp_results_dir / "agent_clusters.csv"
    assert agent_clusters.exists(), f"Missing grouping copy at {agent_clusters}"

    cluster_metrics = exp_results_dir / "metrics" / "ClusterMetrics.csv"
    assert cluster_metrics.exists(), f"Missing cluster metrics at {cluster_metrics}"

    # No observation_split in clusters-eta-mask.json -> one global observation_type.
    with open(exp_results_dir / "exp_config.json", encoding="utf-8") as f:
        dumped = json.load(f)
    assert dumped["observation_split"] is False
    assert dumped["cluster_observation_types"] == {
        "0": "trip_info_eta_mask_norm",
        "1": "trip_info_eta_mask_norm",
    }
    assert len(set(dumped["obs_size_by_cluster"].values())) == 1, (
        "Without observation_split both clusters must share one observation size, "
        f"got {dumped['obs_size_by_cluster']}"
    )


def test_subav_ifppo_execution_with_observation_split(tmp_path):
    """cluster 1 (T set) keeps trip_info_eta_mask_norm while cluster 0 gets raw,
    unnormalized ETAs, so the two clusters end up with different obs sizes."""

    exp_results_dir = run_subav_experiment(SPLIT_EXP_ID, "clusters-eta-split", tmp_path)

    agent_clusters = exp_results_dir / "agent_clusters.csv"
    assert agent_clusters.exists(), f"Missing grouping copy at {agent_clusters}"

    cluster_metrics = exp_results_dir / "metrics" / "ClusterMetrics.csv"
    assert cluster_metrics.exists(), f"Missing cluster metrics at {cluster_metrics}"

    with open(exp_results_dir / "exp_config.json", encoding="utf-8") as f:
        dumped = json.load(f)

    assert dumped["observation_split"] is True
    assert dumped["cluster_observation_types"] == {
        "0": "trip_info_eta",
        "1": "trip_info_eta_mask_norm",
    }

    obs_sizes = dumped["obs_size_by_cluster"]
    assert set(obs_sizes) == {"0", "1"}, f"Both clusters must report a size, got {obs_sizes}"
    assert all(int(size) > 0 for size in obs_sizes.values()), obs_sizes
    assert obs_sizes["0"] != obs_sizes["1"], (
        "trip_info_eta and trip_info_eta_mask_norm must produce different observation "
        f"sizes, both are {obs_sizes}"
    )

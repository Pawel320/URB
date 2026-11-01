"""Per-cluster travel time metrics for SubAV experiments.

Reads the per-episode RouteRL records of an experiment together with the
grouping CSV copy saved by the experiment script (results/<id>/agent_clusters.csv)
and produces:

  - metrics/ClusterMetrics.csv  : per-episode mean travel time for the T set
    (AVs with cluster 1), cluster-0 AVs, all AVs, humans and the whole city.
  - metrics/plots/tt per cluster.png : the same quantities over episodes.

Modeled after analysis/metrics.py; run standalone or via
scripts/utils.py:run_cluster_metrics after a SubAV experiment.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

from metrics import get_episodes
from metrics import plot_vector_values


NUMERIC_COLUMNS = ["travel_time", "id", "action", "origin", "destination", "start_time", "reward"]

MEAN_COLUMNS = [
    "mean_TT_T",
    "mean_TT_cluster0_AV",
    "mean_TT_AV",
    "mean_TT_human",
    "mean_TT_overall",
]


def load_grouping(grouping_path: str) -> dict:
    """Load the agent->cluster mapping from the experiment's agent_clusters.csv."""

    if not os.path.exists(grouping_path):
        raise FileNotFoundError(f"Grouping file not found at: {grouping_path}")

    grouping_df = pd.read_csv(grouping_path)
    cluster_col = next(
        (col for col in ("cluster", "cluster_id") if col in grouping_df.columns),
        None,
    )
    if "agent_id" not in grouping_df.columns or cluster_col is None:
        raise ValueError(
            f"Grouping file must contain 'agent_id' and 'cluster' columns; "
            f"found {list(grouping_df.columns)}"
        )

    return {
        int(row.agent_id): int(getattr(row, cluster_col))
        for row in grouping_df.itertuples(index=False)
    }


def load_episode_records(file: str) -> pd.DataFrame:
    """Load a single episodes/epN.csv file in long (per-agent) format."""

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f"Episode file not found at: {file}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        print(f"Error parsing file: {file}: {e}")
        return pd.DataFrame()

    if df.empty or "id" not in df.columns or "kind" not in df.columns:
        return pd.DataFrame()

    numeric_df = df[[col for col in NUMERIC_COLUMNS if col in df.columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    df[numeric_df.columns] = numeric_df

    return df


def cluster_means_for_episode(df: pd.DataFrame, cluster_map: dict, episode: int, verbose: bool) -> dict:
    """Compute per-cluster mean travel times for one episode DataFrame."""

    if df.empty or "travel_time" not in df.columns:
        return {}

    clusters = df["id"].map(lambda agent_id: cluster_map.get(int(agent_id)))
    is_av = df["kind"] == "AV"
    is_human = df["kind"] == "Human"

    is_t = is_av & (clusters == 1)
    is_c0 = is_av & (clusters == 0)
    unclassified_av = is_av & clusters.isna()

    if unclassified_av.any() and verbose:
        missing_ids = sorted(set(df.loc[unclassified_av, "id"].astype(int)))
        print(
            f"Warning: episode {episode} has {len(missing_ids)} AVs missing from "
            f"the grouping file; examples: {missing_ids[:10]}"
        )

    def mean_tt(mask):
        return float(df.loc[mask, "travel_time"].mean()) if mask.any() else np.nan

    return {
        "episode": episode,
        "n_T": int(is_t.sum()),
        "n_cluster0_AV": int(is_c0.sum()),
        "n_AV": int(is_av.sum()),
        "n_human": int(is_human.sum()),
        "mean_TT_T": mean_tt(is_t),
        "mean_TT_cluster0_AV": mean_tt(is_c0),
        "mean_TT_AV": mean_tt(is_av),
        "mean_TT_human": mean_tt(is_human),
        "mean_TT_overall": mean_tt(pd.Series(True, index=df.index)),
    }


def collect_cluster_metrics(data_path: str, grouping_path: str, verbose: bool = False) -> pd.DataFrame:
    """Collect per-episode cluster metrics for an experiment folder."""

    cluster_map = load_grouping(grouping_path)

    episodes_path = os.path.join(data_path, "episodes")
    episodes = get_episodes(episodes_path)
    if verbose:
        print(f"Loading {len(episodes)} episodes...")

    rows = []
    for episode in episodes:
        episode_file = os.path.join(episodes_path, f"ep{episode}.csv")
        df = load_episode_records(episode_file)
        if df.empty:
            continue
        row = cluster_means_for_episode(df, cluster_map, episode, verbose)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows)
    metrics_df["episode"] = metrics_df["episode"].astype("int32")
    return metrics_df


RESULTS_DEFAULT_DIR = "./results"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-cluster travel time metrics for a SubAV experiment.")
    parser.add_argument("--id", type=str, required=True, help="Experiment ID to process.")
    parser.add_argument("--results-folder", type=str, default=RESULTS_DEFAULT_DIR, help="Root folder where experiment results are stored.")
    parser.add_argument("--verbose", type=bool, default=False, help="Enable verbose output.")

    args = parser.parse_args()

    exp_id = args.id
    results_folder = args.results_folder
    verbose = args.verbose
    if verbose:
        print(f"Experiment ID: {exp_id}")
        print(f"results folder: {results_folder}")

    data_path = ""
    for root, dirs, files in os.walk(results_folder):
        if exp_id in dirs:
            data_path = os.path.join(root, exp_id)
            break

    if data_path == "":
        print(f"Experiment ID {exp_id} not found in {results_folder}")
        sys.exit(1)

    grouping_path = os.path.join(data_path, "agent_clusters.csv")
    if not os.path.exists(grouping_path):
        print(
            f"Grouping file not found at {grouping_path}. "
            "Cluster metrics are only available for SubAV experiments."
        )
        sys.exit(1)

    metrics_path = os.path.join(data_path, "metrics")
    plot_path = os.path.join(metrics_path, "plots")
    cluster_csv_path = os.path.join(metrics_path, "ClusterMetrics.csv")

    try:
        os.makedirs(plot_path, exist_ok=True)
        if verbose:
            print(f"Ensured metrics directory structure exists at: {metrics_path}")
    except OSError as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)

    try:
        metrics_df = collect_cluster_metrics(data_path, grouping_path, verbose)
    except FileNotFoundError as e:
        print(f"Error during cluster metrics collection: {e}")
        sys.exit(1)

    if metrics_df.empty:
        print("No episode data found. Cannot extract cluster metrics.")
        sys.exit(1)

    metrics_df.to_csv(cluster_csv_path, index=False, float_format="%.4f")
    if verbose:
        print(f"Saved cluster metrics to {cluster_csv_path}")

    plot_df = metrics_df[["episode"] + MEAN_COLUMNS].copy()
    if plot_df.empty:
        if verbose:
            print("Cluster metrics are empty. Skipping plots.")
    else:
        if verbose:
            print("Generating plots...")
        plot_vector_values(
            plot_df,
            plot_path,
            "tt per cluster",
            "Mean travel time",
        )
        if verbose:
            print(f"Saved plot to {os.path.join(plot_path, 'tt per cluster.png')}")

        # Mirror the plot next to the environment plots (results/<id>/plots/)
        # so the per-cluster travel times sit beside travel_times.png.
        env_plots_path = os.path.join(data_path, "plots")
        mirrored_plot_path = os.path.join(env_plots_path, "tt per cluster.png")
        try:
            os.makedirs(env_plots_path, exist_ok=True)
            shutil.copy2(os.path.join(plot_path, "tt per cluster.png"), mirrored_plot_path)
            if verbose:
                print(f"Mirrored plot to {mirrored_plot_path}")
        except OSError as e:
            print(f"Warning: could not mirror plot to {mirrored_plot_path}: {e}")

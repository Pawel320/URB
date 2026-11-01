import xml.etree.ElementTree as ET
import os
import csv
import subprocess
import sys
from typing import Optional




class CSVLossLogger:

    def __init__(self, path: str, columns: list[str]):
        self.path = path
        self.columns = columns

        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns, extrasaction='ignore')
        self.writer.writeheader()

    def __call__(self, record: dict)->None:
        self.writer.writerow({column: record.get(column, "") for column in self.columns})
        self.file.flush() # push buffered data to OS immediately

    def close(self):
        # Development note: instead of manual closing, context manager may be added to handle file opening/closing safely using 'with open(...)'
        self.file.close()






import torch
import torch.nn as nn

def get_episodes(ep_path: str) -> list[int]:
    """Get the episodes data

    Returns:
        sorted_episodes (list[int]): the sorted episodes data
    Raises:
        FileNotFoundError: If the episodes folder does not exist
    """

    eps = list()
    if os.path.exists(ep_path):
        for file in os.listdir(ep_path):
            episode = int(file.split("ep")[1].split(".csv")[0])
            eps.append(episode)
    else:
        raise FileNotFoundError(f"Episodes folder does not exist!")


    return sorted(eps)


def clear_SUMO_files(sumo_path, ep_path, remove_additional_files=False):
    '''
        Clear SUMO files that are empty or not in the episodes folder.
        Works only for the consecutive files with the same name.
        The files are named as <file_name>_<episode>.xml

        This is a destructive function, it will remove files from the directory!
    '''
    file_id = 1
    episode = 1

    file_name = "detailed_sumo_stats"
    
    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <tripinfos> is empty (no <tripinfo> elements)
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            if len(root.findall("tripinfo")) == 0:
                # remove the file
                os.remove(file_path)
                # print(f"Removed empty file: {file_path}")
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                # print(f"Renamed file {file_path} to {new_file_path}")
                file_id += 1
        else:
            break
        episode += 1

    file_id = 1
    episode = 1

    file_name = "sumo_stats"

    while True:
        # check if file exists
        file_path = os.path.join(sumo_path, f"{file_name}_{episode}.xml")
        if os.path.exists(file_path):
            # read xml file and check if <vehicle loaded=0>
            try:
                tree = ET.parse(file_path)
            except ET.ParseError:
                print(f"Error parsing XML file: {file_path}")
                break
            root = tree.getroot()
            vehicle = root.find("vehicles")
            if vehicle is not None and vehicle.attrib.get("loaded") == "0":
                # remove the file
                os.remove(file_path)
            else:
                # rename to the next file_id
                new_file_path = os.path.join(sumo_path, f"{file_name}_{file_id}.xml")
                os.rename(file_path, new_file_path)
                file_id += 1
        else:
            break
        episode += 1
    if remove_additional_files:
        episodes = get_episodes(ep_path)
        # remove SUMO files that are not in the episodes
        for file in os.listdir(sumo_path):
            if file.endswith(".xml"):
                episode = int(file.split("_")[-1].split(".")[0])
                if episode not in episodes:
                    os.remove(os.path.join(sumo_path, file))
                    
                    
def print_agent_counts(env):
    print(f"""
    ----------------------------------------------------
                    Agents in traffic
    ----------------------------------------------------
    Total agents           | {len(env.all_agents)}
    Human agents           | {len(env.human_agents)}
    AV agents              | {len(env.machine_agents)}
    ----------------------------------------------------
    """)


def save_loss_records(records_folder: str, loss_records: list[dict], columns: list[str]) -> str:
    """
    Save training loss records to a unified CSV file.

    Args:
        records_folder (str): Experiment output folder (e.g. ../results/<exp_id>).
        loss_records (list[dict]): Row-wise loss data.
        columns (list[str]): Ordered CSV columns.

    Returns:
        str: Absolute path to the saved CSV file.
    """
    losses_folder = os.path.join(records_folder, "losses")
    os.makedirs(losses_folder, exist_ok=True)
    loss_csv_path = os.path.join(losses_folder, "losses.csv")

    with open(loss_csv_path, "w", newline="", encoding="utf-8") as loss_file:
        writer = csv.DictWriter(loss_file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in loss_records:
            writer.writerow({column: row.get(column, "") for column in columns})

    return os.path.abspath(loss_csv_path)


class AppendODEmbedding(nn.Module):
    """
    Append a learned OD embedding to each agent observation.

    The OD ids are fixed for a given experiment and must follow the same order as
    the TorchRL group of machine agents.
    """

    def __init__(self, od_ids: list[int], num_od_pairs: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_od_pairs, embedding_dim)
        self.register_buffer("od_ids", torch.as_tensor(od_ids, dtype=torch.long))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # observation has shape [..., n_agents, obs_dim]
        batch_shape = observation.shape[:-2]
        od_ids = self.od_ids

        if batch_shape:
            od_ids = od_ids.view(*([1] * len(batch_shape)), *od_ids.shape)
            od_ids = od_ids.expand(*batch_shape, *self.od_ids.shape)

        od_embedding = self.embedding(od_ids)
        return torch.cat([observation, od_embedding], dim=-1)


def get_od_ids_for_group(group_agent_ids: list[str], machine_agents: list, num_destinations: int) -> list[int]:
    """
    Build OD ids in the exact order used by the TorchRL group.

    Each OD pair is mapped to one integer:
        od_id = origin * num_destinations + destination
    """

    machine_by_id = {str(machine.id): machine for machine in machine_agents}
    od_ids = []
    for agent_id in group_agent_ids:
        machine = machine_by_id[agent_id]
        od_ids.append(int(machine.origin) * num_destinations + int(machine.destination))
    return od_ids


def script_path_for_config(script_file: str, repo_root: Optional[str] = None) -> str:
    """
    Return a repository-relative script path for exp_config.json when possible.
    Falls back to absolute path if the script is outside of the repository root.
    """
    script_abs = os.path.abspath(script_file)
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        repo_root = os.path.abspath(repo_root)

    try:
        relative_path = os.path.relpath(script_abs, repo_root)
    except ValueError:
        return script_abs

    if relative_path.startswith(".."):
        return script_abs

    return relative_path.replace(os.sep, "/")


def run_metrics_analysis(exp_id: str, results_folder: str = "../results", verbose: bool = False) -> bool:
    """
    Run analysis/metrics.py for a finished experiment.

    The helper is intentionally non-failing so experiment scripts do not crash
    if post-processing fails.
    """
    metrics_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "analysis", "metrics.py")
    )
    results_folder_path = os.path.abspath(results_folder)

    command = [
        sys.executable,
        metrics_script,
        "--id",
        exp_id,
        "--results-folder",
        results_folder_path,
    ]
    if verbose:
        command.extend(["--verbose", "True"])

    print(f"Running metrics analysis for experiment '{exp_id}'...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            f"Warning: Metrics analysis failed for experiment '{exp_id}' "
            f"(exit code {result.returncode})."
        )
        return False
    return True


def run_cluster_metrics(exp_id: str, results_folder: str = "../results", verbose: bool = False) -> bool:
    """
    Run analysis/cluster_metrics.py for a finished SubAV experiment.

    The helper is intentionally non-failing so experiment scripts do not crash
    if post-processing fails (it also exits non-zero when the experiment has
    no agent_clusters.csv grouping file).
    """
    metrics_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "analysis", "cluster_metrics.py")
    )
    results_folder_path = os.path.abspath(results_folder)

    command = [
        sys.executable,
        metrics_script,
        "--id",
        exp_id,
        "--results-folder",
        results_folder_path,
    ]
    if verbose:
        command.extend(["--verbose", "True"])

    print(f"Running cluster metrics analysis for experiment '{exp_id}'...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            f"Warning: Cluster metrics analysis failed for experiment '{exp_id}' "
            f"(exit code {result.returncode})."
        )
        return False
    return True



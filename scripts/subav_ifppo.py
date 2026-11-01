"""SubAV experiment: dual-cluster AV fleet with per-cluster behavior and policies.

The AV fleet is partitioned by an externally provided grouping CSV
(columns 'agent_id' and 'cluster' or 'cluster_id'):
  - cluster 1 = "T" set: collaborative reward (own_TT + T-cluster mean TT),
    trained with its own policy model.
  - cluster 0: selfish reward (own_TT only), trained with its own policy model.
Humans labeled cluster=1 are NOT part of T (T = kind==AV AND cluster==1).

Rewards come exclusively from the environment (RouteRL behavior coefficients);
the script only stamps per-cluster behaviors after mutation and patches
MachineAgent.get_reward so that the collaborative 'group' term covers T
members only. No script-side reward pooling.

Derived from ifppo.py (IF-PPO): two IFPPOV1 models, group-conditioned critic
context for the T set.
"""

import os
import sys

LAUNCH_CWD = os.getcwd()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import ast
import json
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from routerl         import TrafficEnvironment
from tqdm            import tqdm
from collections     import deque, defaultdict

from baseline_models import BaseLearningModel
from iql             import Network
from utils           import clear_SUMO_files
from utils           import print_agent_counts
from utils           import run_cluster_metrics
from utils           import run_metrics_analysis
from utils           import save_loss_records
from utils           import script_path_for_config
from clustered_routes import ClusteredRoutesLoader, resolve_route_set


class IFPPOV1(BaseLearningModel):
    """
    Interaction-Factored PPO v1 for one-step URB routing.

    Actor: shared and decentralized, pi(a_i | o_i).
    Critic: shared, but evaluated on a fixed group-conditioned context built
    from the externally supplied interaction partition.

    Reward and PPO objective are unchanged from the MAPPO baseline.
    """
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        num_agents: int = 20,
        # --- policy settings ---
        shared_policy: bool = False,
        policy_nets: list[nn.Module] | None = None,
        policy_arch_kwargs: dict | None = None,
        # --- critic settings ---
        share_critic: bool = True,
        critic_nets: list[nn.Module] | None = None,
        critic_arch_kwargs: dict | None = None,
        # --- default architecture parameters ---
        default_widths: list[int] = [64,64],
        # --- hyperparameters ---
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        batch_size: int = 64,
        memory_size: int = 5000,
        max_norm: float = 0.5,
        device: torch.device | None = None,
        action_mask: dict | None = None,
        central_state_size: int | None = None,
        **kwargs
    ):
        super().__init__()
        # device setup 
        self.device = device or torch.device("cpu")
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.num_agents = num_agents
        self.max_norm = max_norm
        self.central_state_size = (
        central_state_size
            if central_state_size is not None
            else state_size
        )

        self.last_central_states = {}
        self.action_masks = action_mask if action_mask is not None else {}
        self.num_epochs = kwargs.get("num_epochs", 3)
        ws = kwargs.get("widths", default_widths)
        self.clip_ratio = kwargs.get("clip_eps", clip_ratio)
        lr_actor = kwargs.get("lr", lr_actor)
        lr_critic = kwargs.get("lr", lr_critic)

        # training phase flag
        self.training = True
        
        # hyperparameters
        self.gamma = gamma
        self.entropy_coef = kwargs.get("entropy_coef", entropy_coef)
        self.value_coef = kwargs.get("value_coef", value_coef)
        self.batch_size = kwargs.get("batch_size", batch_size)
        self.memory = []
        self.last_states = {}
        self.last_actions = {}
        self.last_log_probs = {}

        # --- Policy networks ---
        if policy_nets is not None:
            assert len(policy_nets) == num_agents or shared_policy, \
                "If policy_nets is provided, it must match the number of agents or be shared."
            self.policies = [net.to(self.device) for net in (policy_nets if not shared_policy else [policy_nets[0]] * num_agents)]
        else:
            self.policies = []
            for _ in range(num_agents):
                net = Network(state_size, action_space_size, len(ws) - 1, ws).to(self.device)
                self.policies.append(net)
            if shared_policy:
                self.policies = [self.policies[0]] * num_agents

        # create actor optimizers
        if shared_policy:
            self.actor_optimizer = optim.Adam(self.policies[0].parameters(), lr=lr_actor)
        else:
            self.actor_optimizer = [optim.Adam(policy.parameters(), lr=lr_actor) for policy in self.policies]
        
        # --- Critic networks ---
        if critic_nets is not None:
            assert len(critic_nets) == num_agents or share_critic, \
                "If critic_nets is provided, it must match the number of agents or be shared."
            self.critics = [net.to(self.device) for net in (critic_nets if not share_critic else [critic_nets[0]] * num_agents)]
        else:
            # build critics using generic Network class
            ch_ws = critic_arch_kwargs.get('widths', default_widths) if critic_arch_kwargs else default_widths
            self.critics = []
            for _ in range(num_agents):
                net = Network(
                    self.central_state_size,
                    1,
                    len(ch_ws) - 1,
                    ch_ws
                ).to(self.device)

                self.critics.append(net)
            if share_critic:
                shared_critic = self.critics[0]
                self.critics = [shared_critic] * num_agents

        # create critic optimizers
        if share_critic:
            self.critic_optim = optim.Adam(self.critics[0].parameters(), lr=lr_critic)
        else:
            self.critic_optim = [optim.Adam(net.parameters(), lr=lr_critic) for net in self.critics]

        # loss / PPO diagnostics tracking
        self.loss_actor = []
        self.loss_critic = []
        self.diagnostics = []
        self.update_count = 0

    def train(self):
        """Set all models to training mode"""
        for policy in self.policies:
            policy.train()
        for critic in self.critics:
            critic.train()
        self.training = True
        return self
    
    def eval(self):
        """Set all models to evaluation mode"""
        for policy in self.policies:
            policy.eval()
        for critic in self.critics:
            critic.eval()
        self.training = False
        return self

    def push(self, agent_id: int | float, reward: float | None = None, next_state=None, done: bool = True):
        if reward is None:
            reward = float(agent_id)
            agent_id = 0
        if agent_id not in self.last_states:
            return
        if next_state is None:
            next_state = self.last_states[agent_id]
        
        self.memory.append(
        (
            self.last_states.pop(agent_id),
            self.last_central_states.pop(agent_id),
            self.last_actions.pop(agent_id),
            float(reward),
            self.last_log_probs.pop(agent_id),
            next_state,
            bool(done),
            agent_id,
            )
        )

    def act(self, state: any, agent_id: int, central_state=None):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policies[agent_id](state_tensor)
            
            if agent_id in self.action_masks:
                mask = self.action_masks[agent_id]
                logits = logits.masked_fill(~mask, float('-inf'))
                
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item() if self.training else torch.argmax(logits).item()
            
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        if central_state is None:
            central_state = state

        self.last_central_states[agent_id] = np.asarray(
        central_state,
        dtype=np.float32
        )

        self.last_states[agent_id] = state
        self.last_actions[agent_id] = action
        self.last_log_probs[agent_id] = log_prob

        return action

    def learn(self):
        n_samples = len(self.memory)
        if not self.memory:
            return None
    
        (
            s_batch,
            cs_batch,
            a_batch,
            r_batch,
            lp_batch,
            ns_batch,
            d_batch,
            id_batch,
        ) = zip(*self.memory)

        states_tensor = torch.FloatTensor(np.array(s_batch)).to(self.device)
        actions_tensor = torch.LongTensor(a_batch).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(ns_batch)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(lp_batch).unsqueeze(1).to(self.device)
        dones_tensor = torch.FloatTensor(d_batch).unsqueeze(1).to(self.device)
        id_tensor = torch.LongTensor(id_batch).to(self.device)

        central_states_tensor = torch.FloatTensor(
            np.array(cs_batch)
        ).to(self.device)

        unique_ids = id_tensor.unique().tolist()

        advantages = torch.zeros_like(rewards_tensor)
        targets = torch.zeros_like(rewards_tensor)

        with torch.no_grad():
            for aid in unique_ids:
                mask = (id_tensor == aid)
                if not mask.any(): continue
                
                v = self.critics[aid](central_states_tensor[mask])
                
                td_target = rewards_tensor[mask]
                adv = td_target - v
                
                advantages[mask] = adv
                targets[mask] = td_target

        # PPO update diagnostics, aggregated sample-weighted across minibatches/epochs.
        diag_weight = 0
        diag_actor_loss = 0.0
        diag_policy_loss = 0.0
        diag_critic_loss = 0.0
        diag_entropy = 0.0
        diag_approx_kl = 0.0
        diag_clip_fraction = 0.0

        for _ in range(self.num_epochs):
            indices = torch.randperm(len(self.memory), device=self.device)

            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start : start + self.batch_size]

                mb_states = states_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_log_probs = old_log_probs_tensor[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_targets = targets[mb_idx]
                mb_central_states = central_states_tensor[mb_idx]
                mb_ids = id_tensor[mb_idx]
                mb_unique_ids = mb_ids.unique().tolist()

                avg_policy_loss, avg_critic_loss, avg_entropy = 0.0, 0.0, 0.0
                avg_approx_kl, avg_clip_fraction = 0.0, 0.0
                total_count = len(mb_idx)

                for aid in mb_unique_ids:
                    agent_mask = (mb_ids == aid)
                    agent_count = agent_mask.sum().item()

                    #critic
                    v = self.critics[aid](mb_central_states[agent_mask])
                    critic_loss = nn.MSELoss()(v, mb_targets[agent_mask])

                    #actor
                    logits = self.policies[aid](mb_states[agent_mask])
                    
                    if aid in self.action_masks:
                        action_mask = self.action_masks[aid]
                        logits = logits.masked_fill(~action_mask, float('-inf'))
                        
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions[agent_mask].squeeze(1)).unsqueeze(1)
                    entropy = dist.entropy().mean()

                    log_ratio = new_log_probs - mb_old_log_probs[agent_mask]
                    ratios = torch.exp(log_ratio)
                    clipped_ratios = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)

                    surr1 = ratios * mb_advantages[agent_mask]
                    surr2 = clipped_ratios * mb_advantages[agent_mask]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    with torch.no_grad():
                        approx_kl = ((ratios - 1.0) - log_ratio).mean()
                        clip_fraction = (
                            torch.abs(ratios - 1.0) > self.clip_ratio
                        ).float().mean()

                    weight = agent_count / total_count
                    avg_critic_loss += critic_loss * weight
                    avg_policy_loss += policy_loss * weight
                    avg_entropy += entropy * weight
                    avg_approx_kl += approx_kl * weight
                    avg_clip_fraction += clip_fraction * weight

                if isinstance(self.critic_optim, list):
                    for aid in mb_unique_ids: self.critic_optim[aid].zero_grad()
                    avg_critic_loss.backward()
                    for aid in mb_unique_ids: self.critic_optim[aid].step()
                else:
                    self.critic_optim.zero_grad()
                    avg_critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics[0].parameters(),
                        max_norm=self.max_norm
                    )
                    self.critic_optim.step()

                total_loss = avg_policy_loss - self.entropy_coef * avg_entropy
                if isinstance(self.actor_optimizer, list):
                    for aid in mb_unique_ids: self.actor_optimizer[aid].zero_grad()
                    total_loss.backward()
                    for aid in mb_unique_ids: self.actor_optimizer[aid].step()
                else:
                    self.actor_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policies[0].parameters(),
                        max_norm=self.max_norm
                    )
                    self.actor_optimizer.step()

                self.loss_critic.append(avg_critic_loss.item())
                self.loss_actor.append(avg_policy_loss.item())

                diag_weight += total_count
                diag_actor_loss += float(total_loss.detach().item()) * total_count
                diag_policy_loss += float(avg_policy_loss.detach().item()) * total_count
                diag_critic_loss += float(avg_critic_loss.detach().item()) * total_count
                diag_entropy += float(avg_entropy.detach().item()) * total_count
                diag_approx_kl += float(avg_approx_kl.detach().item()) * total_count
                diag_clip_fraction += float(avg_clip_fraction.detach().item()) * total_count

        self.update_count += 1
        denom = max(diag_weight, 1)
        self.diagnostics.append(
            {
                "update": self.update_count,
                "samples": n_samples,
                "loss": diag_actor_loss / denom,
                "policy_loss": diag_policy_loss / denom,
                "critic_loss": diag_critic_loss / denom,
                "entropy": diag_entropy / denom,
                "approx_kl": diag_approx_kl / denom,
                "clip_fraction": diag_clip_fraction / denom,
                "reward_mean": float(rewards_tensor.mean().item()),
                "reward_std": float(rewards_tensor.std(unbiased=False).item()),
            }
        )

        self.memory.clear()
        return self.diagnostics[-1]

    def get_last_observation(self, agent_id: int):
        return self.last_states.get(agent_id, None)
    
    def get_last_action(self, agent_id: int):
        return self.last_actions.get(agent_id, None)
    
    def get_last_log_prob(self, agent_id: int):
        return self.last_log_probs.get(agent_id, None)
    
    def get_policy(self, agent_id: int):
        return self.policies[agent_id] if agent_id < len(self.policies) else None


def install_cluster_aware_reward():
    """Patch MachineAgent.get_reward so the behavior 'group' term covers only
    same-cluster AVs for cluster-stamped agents.

    The stamping (agent.cluster_id / agent.cluster_map) is done by this script
    after mutation; unstamped agents keep the stock RouteRL behavior (group =
    same kind). Patched at runtime so RouteRL-URB itself stays untouched.
    """
    from routerl.keychain import Keychain as kc
    from routerl.environment.agent import MachineAgent

    if getattr(MachineAgent, "_subav_cluster_reward_patched", False):
        return
    original_get_reward = MachineAgent.get_reward

    def cluster_aware_get_reward(self, observation, group_vicinity=False):
        cluster_map = getattr(self, "cluster_map", None)
        cluster_id = getattr(self, "cluster_id", None)
        if cluster_map is None or cluster_id is None:
            return original_get_reward(self, observation, group_vicinity)

        vicinity_obs = list()
        if group_vicinity == True:
            min_start_time, max_start_time = self.start_time - self.observed_span, self.start_time + self.observed_span
            for obs in observation:
                if (obs[kc.AGENT_ORIGIN], obs[kc.AGENT_DESTINATION]) == (self.origin, self.destination):
                    if min_start_time <= obs[kc.AGENT_START_TIME] <= max_start_time:
                        vicinity_obs.append(obs)
        else:
            for obs in observation:
                vicinity_obs.append(obs)

        group_obs, others_obs, all_obs, own_tt = list(), list(), list(), None
        for obs in vicinity_obs:
            all_obs.append(obs[kc.TRAVEL_TIME])
            if obs[kc.AGENT_KIND] == self.kind:
                if cluster_map.get(obs[kc.AGENT_ID]) == cluster_id:
                    group_obs.append(obs[kc.TRAVEL_TIME])
            else:
                others_obs.append(obs[kc.TRAVEL_TIME])
            if obs[kc.AGENT_ID] == self.id:
                own_tt = obs[kc.TRAVEL_TIME]

        group_tt = np.mean(group_obs) if group_obs else 0
        others_tt = np.mean(others_obs) if others_obs else 0
        all_tt = np.mean(all_obs) if all_obs else 0

        a, b, c, d = self.rewards_coefs
        agent_reward = a * own_tt + b * group_tt + c * others_tt + d * all_tt
        return agent_reward

    MachineAgent.get_reward = cluster_aware_get_reward
    MachineAgent._subav_cluster_reward_patched = True


def load_grouping(grouping_path):
    """Load and validate the agent grouping CSV (agent_id + cluster columns)."""
    grouping_df = pd.read_csv(grouping_path)

    cluster_col = None
    for candidate in ("cluster", "cluster_id"):
        if candidate in grouping_df.columns:
            cluster_col = candidate
            break
    if cluster_col is None or "agent_id" not in grouping_df.columns:
        raise ValueError(
            f"Grouping CSV must contain 'agent_id' and 'cluster' (or 'cluster_id') columns; "
            f"found {list(grouping_df.columns)}"
        )

    grouping_df = grouping_df[["agent_id", cluster_col]].rename(columns={cluster_col: "cluster"})
    if grouping_df[["agent_id", "cluster"]].isna().any().any():
        raise ValueError("Grouping CSV contains missing agent_id/cluster values.")

    grouping_df["agent_id"] = grouping_df["agent_id"].astype(int)
    grouping_df["cluster"] = grouping_df["cluster"].astype(int)

    if grouping_df["agent_id"].duplicated().any():
        duplicate_ids = (
            grouping_df.loc[grouping_df["agent_id"].duplicated(keep=False), "agent_id"]
            .unique()
            .tolist()
        )
        raise ValueError(
            "Grouping CSV contains duplicate agent IDs: "
            + ", ".join(map(str, duplicate_ids[:20]))
        )

    invalid_labels = sorted(set(grouping_df["cluster"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"Grouping CSV cluster labels must be 0 or 1, found: {invalid_labels}"
        )

    return grouping_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, required=True)
    parser.add_argument('--env-conf', type=str, default="clusters-eta-mask")
    parser.add_argument('--task-conf', type=str, required=True)
    parser.add_argument('--alg-conf', type=str, required=True)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--env-seed', type=int, default=42)
    parser.add_argument('--torch-seed', type=int, default=42)
    parser.add_argument(
            '--route-set',
            type=str,
            default=None,
            help="Named route-set subdirectory. Uses the network default when omitted.",
        )
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help=(
            "Root folder for experiment results (default: ../results next to this script, "
            "i.e. URB/results). Absolute path or resolved from the launch directory."
        ),
    )
    parser.add_argument(
        '--grouping-file',
        type=str,
        default=None,
        help=(
            "CSV with columns 'agent_id' and 'cluster' (or 'cluster_id'), labels 0/1. "
            "Optional if 'grouping_file' is set in a config. Relative paths are resolved "
            "from the directory where this script was launched."
        ),
    )
    args = parser.parse_args()

    CONFIG_ALGORITHM = "ifppo"
    ALGORITHM = "subav_ifppo"
    exp_id = args.id
    alg_config = args.alg_conf
    env_config = args.env_conf
    task_config = args.task_conf
    network = args.net
    env_seed = args.env_seed
    torch_seed = args.torch_seed
    requested_route_set = args.route_set
    shuffle = args.shuffle

    grouping_path = args.grouping_file
    if grouping_path is not None:
        if not os.path.isabs(grouping_path):
            grouping_path = os.path.abspath(os.path.join(LAUNCH_CWD, grouping_path))
        if not os.path.exists(grouping_path):
            raise FileNotFoundError(f"Grouping CSV not found: {grouping_path}")

    print("### STARTING EXPERIMENT ###")
    print(f"Algorithm: {ALGORITHM.upper()}")
    print(f"Experiment ID: {exp_id}")
    print(f"Network: {network}")
    print(f"Environment seed: {env_seed}")
    print(f"Algorithm config: {alg_config}")
    print(f"Environment config: {env_config}")
    print(f"Task config: {task_config}")
    print(f"Requested route set: {requested_route_set or 'network default'}")
    print(f"Shuffle: {shuffle}")
    print(f"Agent grouping: {grouping_path or '(from config)'}")

    if args.results_dir is not None:
        results_root = args.results_dir
        if not os.path.isabs(results_root):
            results_root = os.path.abspath(os.path.join(LAUNCH_CWD, results_root))
    else:
        results_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "results"))
    print(f"Results root: {results_root}")

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(env_seed)
    np.random.seed(env_seed)

    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Device is: ", device)

    install_cluster_aware_reward()

    params = dict()
    alg_params = json.load(open(f"../config/algo_config/{CONFIG_ALGORITHM}/{alg_config}.json"))
    env_params = json.load(open(f"../config/env_config/{env_config}.json"))
    task_params = json.load(open(f"../config/task_config/{task_config}.json"))
    params.update(alg_params)
    params.update(env_params)
    params.update(task_params)
    del params["desc"], env_params, task_params

    if grouping_path is None:
        config_grouping = params.get("grouping_file")
        if not config_grouping:
            raise ValueError(
                "No grouping CSV provided. Pass --grouping-file or set "
                "'grouping_file' in a config."
            )
        grouping_path = config_grouping
        if not os.path.isabs(grouping_path):
            grouping_path = os.path.abspath(os.path.join(LAUNCH_CWD, grouping_path))
        if not os.path.exists(grouping_path):
            raise FileNotFoundError(f"Grouping CSV not found: {grouping_path}")
        print(f"Agent grouping (from config): {grouping_path}")

    observation_type = params.get(
        "observation_type",
        params.get("observations", "previous_agents_plus_start_time"),
    )
    path_gen_workers_value = params.get("path_gen_workers", 4)

    cluster_behaviors_raw = params.get("cluster_behaviors")
    if cluster_behaviors_raw is None:
        default_behavior = params.get("av_behavior", "selfish")
        cluster_behaviors = {"0": default_behavior, "1": default_behavior}
    else:
        cluster_behaviors = {str(key): value for key, value in cluster_behaviors_raw.items()}
    for cluster_label in ("0", "1"):
        if cluster_label not in cluster_behaviors:
            raise ValueError(
                f"cluster_behaviors must define both '0' and '1'; missing '{cluster_label}'."
            )

    use_clustered_routes = params.get("use_clustered_routes", False)
    route_set = (
        resolve_route_set(network, requested_route_set)
        if use_clustered_routes
        else None
    )
    print(f"Route set: {route_set or 'none (unclustered)'}")

    # set params as variables in this script
    for key, value in params.items():
        globals()[key] = value
        
    custom_network_folder = f"../networks/{network}"
    phases = [1, human_learning_episodes, int(training_eps) + human_learning_episodes]
    phase_names = ["Human stabilization", "Mutation and AV learning", "Testing phase"]
    records_folder = os.path.join(results_root, exp_id)
    plots_folder = os.path.join(records_folder, "plots")

    od_file_path = os.path.join(custom_network_folder, f"od_{network}.txt")
    with open(od_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    data = ast.literal_eval(content)
    origins = data['origins']
    destinations = data['destinations']

    agents_csv_path = os.path.join(custom_network_folder, "agents.csv")
    num_agents = len(pd.read_csv(agents_csv_path))
    if os.path.exists(agents_csv_path):
        os.makedirs(records_folder, exist_ok=True)
        new_agents_csv_path = os.path.join(records_folder, "agents.csv")
        with open(agents_csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(new_agents_csv_path, 'w', encoding='utf-8') as f:
            f.write(content)
        max_start_time = pd.read_csv(new_agents_csv_path)['start_time'].max()
    else:
        raise FileNotFoundError(f"Agents CSV file not found at {agents_csv_path}.")
            
    num_machines = int(num_agents * ratio_machines)
    total_episodes = human_learning_episodes + training_eps + test_eps
    
    # Dump exp config to records
    exp_config_path = os.path.join(records_folder, "exp_config.json")
    dump_config = params.copy()
    
    # Load pre-generated clustered routes and their per-OD action masks.
    configured_number_of_paths = params.get("number_of_paths", 4)
    number_of_paths = configured_number_of_paths
    create_paths_flag = True
    action_masks = None
    
    if use_clustered_routes:
        try:
            route_set_dir = os.path.join(custom_network_folder, "clustered_routes", route_set)
            clustered_loader = ClusteredRoutesLoader(
                network,
                custom_network_folder,
                shuffle,
                env_seed,
                route_set_dir=route_set_dir,
            )
            number_of_paths = clustered_loader.get_number_of_paths()
            clustered_loader.export_paths_routes(records_folder, origins, destinations)
            action_masks = clustered_loader.create_masks(origins, destinations)
            if not action_masks:
                raise ValueError("The clustered route set contains no action masks.")

            for od_pair, mask in action_masks.items():
                mask_array = np.asarray(mask)
                if mask_array.shape != (number_of_paths,):
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} has shape "
                        f"{mask_array.shape}; expected ({number_of_paths},)."
                    )
                if not np.isin(mask_array, (0, 1)).all():
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} must be binary."
                    )
                if not mask_array.any():
                    raise ValueError(
                        f"Action mask for OD pair {od_pair} has no valid actions."
                    )

            agent_ods = {
                (int(row.origin), int(row.destination))
                for row in pd.read_csv(
                    agents_csv_path,
                    usecols=["origin", "destination"],
                ).itertuples()
            }
            missing_ods = sorted(agent_ods.difference(action_masks))
            if missing_ods:
                raise ValueError(
                    "Missing action masks for agent OD pairs: "
                    + ", ".join(map(str, missing_ods))
                )

            create_paths_flag = False
            dump_config["number_of_paths"] = number_of_paths
        except FileNotFoundError as e:
            use_clustered_routes = False
            number_of_paths = configured_number_of_paths
            print(f"[CLUSTERED ROUTES] Warning: {e}")
            print("[CLUSTERED ROUTES] Falling back to JanuX generation\n")

    dump_config["network"] = network
    dump_config["env_seed"] = env_seed
    dump_config["torch_seed"] = torch_seed
    dump_config["route_set"] = route_set
    dump_config["env_config"] = env_config
    dump_config["task_config"] = task_config
    dump_config["alg_config"] = alg_config
    dump_config["script"] = script_path_for_config(__file__)
    dump_config["algorithm"] = ALGORITHM
    dump_config["num_agents"] = num_agents
    dump_config["num_machines"] = num_machines
    dump_config["use_clustered_routes"] = use_clustered_routes
    dump_config["use_action_masks"] = action_masks is not None
    dump_config["shuffle"] = shuffle
    dump_config["observation_type"] = observation_type
    dump_config["path_gen_workers"] = path_gen_workers_value
    dump_config["grouping_file"] = grouping_path
    dump_config["cluster_behaviors"] = cluster_behaviors
    dump_config["t_reward_pooling"] = False
    dump_config["t_group_reward_scope"] = "t_cluster_only"
    dump_config["critic_scope"] = "balanced_interaction_group_v1"
    dump_config["critic_context"] = (
        "local observation + same-OD route fractions within assigned group + group progress"
    )
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    wandb.init(
        entity="interaction-aware-marl",
        project="subav",
        name=exp_id,
        config=dump_config
    )


    # Initialize the environment
    env = TrafficEnvironment(
        seed = env_seed,
        create_agents = False,
        create_paths = create_paths_flag,
        action_masks = action_masks,
        save_detectors_info = False,
        agent_parameters = {
            "new_machines_after_mutation": num_machines, 
            "human_parameters": {
                "model": human_model,
                "alpha": human_alpha,
                "beta": human_beta,
                "beta_randomness": human_beta_randomness,
                "deterministic": human_deterministic,
            },
            "machine_parameters" : {
                "behavior" : av_behavior,
                "observation_type" : observation_type
            }
        },
        environment_parameters = {
            "save_every" : save_every,
        },
        simulator_parameters = {
            "network_name" : network,
            "custom_network_folder" : custom_network_folder,
            "sumo_type" : "sumo",
            "simulation_timesteps" : max_start_time
        }, 
        plotter_parameters = {
            "phases" : phases,
            "phase_names" : phase_names,
            "smooth_by" : smooth_by,
            "plot_choices" : plot_choices,
            "records_folder" : records_folder,
            "plots_folder" : plots_folder
        },
        path_generation_parameters = {
            "origins" : origins,
            "destinations" : destinations,
            "number_of_paths" : number_of_paths,
            "beta" : path_gen_beta,
            "num_samples" : num_samples,
            "path_gen_workers" : path_gen_workers_value,
            "visualize_paths" : False
        } 
    )

    env.start()
    env.reset()
    print_agent_counts(env)

    ### Human learning phase ###
    pbar = tqdm(total=total_episodes, desc="Human learning")
    for episode in range(human_learning_episodes):
        env.step()
        pbar.update()
    pbar.close()

    # Mutation
    env.mutation(disable_human_learning = not should_humans_adapt, mutation_start_percentile = -1)
    print_agent_counts(env)
    obs_size = env.observation_space(env.possible_agents[0]).shape[0]

    
    # Set policies for machine agents
    shared_action_space_size = max(agent.action_space_size for agent in env.machine_agents)
    central_extra_size = shared_action_space_size + 1

    central_state_size = (
    obs_size + central_extra_size
    )
    agent_to_idx = {str(agent.id): idx for idx, agent in enumerate(env.machine_agents)}

    # ------------------------------------------------------------------
    # Load fixed cluster groups (0 = background AVs, 1 = T set).
    # ------------------------------------------------------------------
    grouping_df = load_grouping(grouping_path)

    external_agent_to_cluster = {
        str(int(row.agent_id)): int(row.cluster)
        for row in grouping_df.itertuples(index=False)
    }

    machine_ids = [str(agent.id) for agent in env.machine_agents]
    missing_machine_groups = [
        aid for aid in machine_ids if aid not in external_agent_to_cluster
    ]
    if missing_machine_groups:
        raise ValueError(
            f"{len(missing_machine_groups)} machine agents are missing from the grouping CSV. "
            "Examples: " + ", ".join(missing_machine_groups[:20])
        )

    agent_to_cluster = np.array(
        [external_agent_to_cluster[aid] for aid in machine_ids],
        dtype=np.int64,
    )
    num_clusters = 2
    cluster_sizes = np.bincount(agent_to_cluster, minlength=num_clusters).astype(np.int64)
    if np.any(cluster_sizes <= 0):
        empty = [int(i) for i, size in enumerate(cluster_sizes) if size <= 0]
        raise RuntimeError(
            f"Cluster(s) {empty} contain no machine agents; both clusters must be "
            "non-empty among the AVs (T = AVs with cluster_id 1)."
        )

    active_labels = sorted(set(int(label) for label in agent_to_cluster))
    missing_behaviors = [label for label in active_labels if str(label) not in cluster_behaviors]
    if missing_behaviors:
        raise ValueError(
            f"cluster_behaviors is missing entries for active cluster(s) {missing_behaviors}."
        )

    print(f"Active SubAV clusters: {num_clusters}")
    print(
        "Active machine-agent cluster sizes: "
        f"T (cluster 1) = {int(cluster_sizes[1])}, "
        f"cluster 0 = {int(cluster_sizes[0])}"
    )

    # Save the exact grouping used by this experiment for reproducibility
    # and for the post-run cluster metrics.
    grouping_df.to_csv(
        os.path.join(records_folder, "agent_clusters.csv"),
        index=False,
    )

    dump_config["num_active_clusters"] = int(num_clusters)
    dump_config["active_cluster_sizes"] = [int(x) for x in cluster_sizes.tolist()]
    with open(exp_config_path, 'w', encoding='utf-8') as f:
        json.dump(dump_config, f, indent=4)

    # Stamp per-cluster behavior and cluster membership on the machine agents.
    machine_cluster_map = {
        int(agent.id): int(external_agent_to_cluster[str(agent.id)])
        for agent in env.machine_agents
    }
    for agent in env.machine_agents:
        cluster_label = machine_cluster_map[int(agent.id)]
        agent.cluster_id = cluster_label
        agent.cluster_map = machine_cluster_map
        agent.behavior = cluster_behaviors[str(cluster_label)]
        agent.rewards_coefs = agent._get_reward_coefs()
    print(
        "Per-cluster behaviors: "
        + ", ".join(
            f"cluster {label} -> {cluster_behaviors[str(label)]}"
            for label in active_labels
        )
    )

    internal_action_masks = {}
    for idx, agent in enumerate(env.machine_agents):
        if action_masks is not None:
            key = (agent.origin, agent.destination)
            if key not in action_masks:
                key = (int(agent.origin), int(agent.destination))
                
            if key not in action_masks:
                raise ValueError(f"Missing action mask for agent {agent.id}")
                
            valid_mask = action_masks[key]
            mask_tensor = torch.as_tensor(valid_mask, dtype=torch.bool, device=device)
            
            padded_mask = torch.zeros(shared_action_space_size, dtype=torch.bool, device=device)
            limit = min(mask_tensor.size(0), shared_action_space_size)
            padded_mask[:limit] = mask_tensor[:limit]
            
            internal_action_masks[idx] = padded_mask
        else:
            mask_tensor = torch.zeros(shared_action_space_size, dtype=torch.bool, device=device)
            mask_tensor[:agent.action_space_size] = True
            internal_action_masks[idx] = mask_tensor

    model_params = params.copy()
    model_params.update({
        "state_size": obs_size,
        "central_state_size": central_state_size,
        "action_space_size": shared_action_space_size,
        "num_agents": len(env.machine_agents),
        "shared_policy": True,
        "share_critic": True,
        "device": device,
        "action_mask": internal_action_masks
    })
    
    selfish_params = model_params.copy()
    selfish_params["central_state_size"] = obs_size 

    # Two independent models: cooperative (T set) vs selfish (cluster 0).
    coop_model = IFPPOV1(**model_params)
    selfish_model = IFPPOV1(**selfish_params)

    coop_agents_set = set(
        str(agent.id) for agent in env.machine_agents
        if machine_cluster_map[int(agent.id)] == 1
    )

    for agent in env.machine_agents:
        if str(agent.id) in coop_agents_set:
            agent.model = coop_model
        else:
            agent.model = selfish_model
            
    agent_lookup = {str(agent.id): agent for agent in env.machine_agents}

    ### Learning phase ###
    pbar = tqdm(total=training_eps, desc="AV learning")
    os.makedirs(plots_folder, exist_ok=True)
    for episode in range(training_eps):
        # Critic context is strictly group-local:
        #   - route fractions from PREVIOUS agents in the same group AND OD
        #   - progress through the current agent's own group
        cluster_od_route_counts = defaultdict(
            lambda: np.zeros(
                shared_action_space_size,
                dtype=np.float32,
            )
        )
        cluster_num_acted = np.zeros(
            num_clusters,
            dtype=np.int64,
        )

        env.reset()
        episode_rewards = []
        episode_travel_times = []
        ep_rewards_c0, ep_rewards_c1 = [], []
        ep_tt_c0, ep_tt_c1 = [], []
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            str_id = str(agent_id)
            is_coop = str_id in coop_agents_set
            
            acted_od = None
            action = None
            
            if termination or truncation:
                if agent_id in agent_lookup:
                    internal_id = agent_to_idx[str_id]
                    model = coop_model if is_coop else selfish_model
                    model.push(agent_id=internal_id, reward=reward)
                episode_rewards.append(reward)
                if "travel_time" in info:
                    episode_travel_times.append(info["travel_time"])
                if is_coop:
                    ep_rewards_c1.append(reward)
                    if "travel_time" in info:
                        ep_tt_c1.append(info["travel_time"])
                else:
                    ep_rewards_c0.append(reward)
                    if "travel_time" in info:
                        ep_tt_c0.append(info["travel_time"])
            else:
                if agent_id in agent_lookup:
                    internal_id = agent_to_idx[str_id]
                    machine = agent_lookup[agent_id]
                    group_id = int(agent_to_cluster[internal_id])
                    
                    acted_od = (int(machine.origin), int(machine.destination))
                    group_od_key = (group_id, acted_od)

                    if is_coop:
                        counts = cluster_od_route_counts[group_od_key]
                        group_od_route_fractions = (counts / counts.sum()).astype(np.float32) if counts.sum() > 0 else np.zeros_like(counts, dtype=np.float32)
                        group_progress = np.array([cluster_num_acted[group_id] / max(int(cluster_sizes[group_id]), 1)], dtype=np.float32)
                        
                        group_state = np.concatenate([np.asarray(observation, dtype=np.float32), group_od_route_fractions, group_progress]).astype(np.float32)
                        action = coop_model.act(observation, agent_id=internal_id, central_state=group_state)
                    else:
                        # Background AVs see only the local observation
                        action = selfish_model.act(observation, agent_id=internal_id, central_state=observation)

                if acted_od is not None and action is not None and is_coop:
                    cluster_od_route_counts[group_od_key][int(action)] += 1
                    cluster_num_acted[group_id] += 1
                
            env.step(action)

        # Update both models
        if (episode + 1) % update_every == 0:
            coop_model.learn()
            selfish_model.learn()

        metrics = {
            "episode": human_learning_episodes + episode,
            "train/loss_coop": coop_model.diagnostics[-1]["loss"] if coop_model.diagnostics else None,
            "train/loss_selfish": selfish_model.diagnostics[-1]["loss"] if selfish_model.diagnostics else None,
            "train/reward_mean": float(np.mean(episode_rewards)) if episode_rewards else None,
            "train/travel_time_mean": float(np.mean(episode_travel_times)) if episode_travel_times else None,
            "train/reward_mean_c1": float(np.mean(ep_rewards_c1)) if ep_rewards_c1 else None,
            "train/reward_mean_c0": float(np.mean(ep_rewards_c0)) if ep_rewards_c0 else None,
            "train/tt_mean_c1": float(np.mean(ep_tt_c1)) if ep_tt_c1 else None,
            "train/tt_mean_c0": float(np.mean(ep_tt_c0)) if ep_tt_c0 else None,
        }
        wandb.log({k: v for k, v in metrics.items() if v is not None})

        if episode % plot_every == 0:
            env.plot_results()
        pbar.update()
    pbar.close()

    ### Testing phase ###
    coop_model.eval()
    selfish_model.eval()

    pbar = tqdm(total=test_eps, desc="Test phase")
    for episode in range(test_eps):
        env.reset()
        episode_rewards = []
        episode_travel_times = []
        ep_rewards_c0, ep_rewards_c1 = [], []
        ep_tt_c0, ep_tt_c1 = [], []
        
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if agent_id not in agent_lookup or termination or truncation:
                action = None
                if agent_id in agent_lookup:
                    episode_rewards.append(reward)
                    if "travel_time" in info:
                        episode_travel_times.append(info["travel_time"])
                    if is_coop:
                        ep_rewards_c1.append(reward)
                        if "travel_time" in info:
                            ep_tt_c1.append(info["travel_time"])
                    else:
                        ep_rewards_c0.append(reward)
                        if "travel_time" in info:
                            ep_tt_c0.append(info["travel_time"])
            else:
                internal_id = agent_to_idx[str(agent_id)]
                is_coop = str(agent_id) in coop_agents_set
                model = coop_model if is_coop else selfish_model
                action = model.act(observation, agent_id=internal_id)
                
            env.step(action)
            
        wandb.log(
            {
                "episode": human_learning_episodes + training_eps + episode,
                "testing/reward_mean": float(np.mean(episode_rewards)) if episode_rewards else None,
                "testing/travel_time_mean": float(np.mean(episode_travel_times)) if episode_travel_times else None,
                "testing/reward_mean_c1": float(np.mean(ep_rewards_c1)) if ep_rewards_c1 else None,
                "testing/reward_mean_c0": float(np.mean(ep_rewards_c0)) if ep_rewards_c0 else None,
                "testing/tt_mean_c1": float(np.mean(ep_tt_c1)) if ep_tt_c1 else None,
                "testing/tt_mean_c0": float(np.mean(ep_tt_c0)) if ep_tt_c0 else None,
            },
            step=human_learning_episodes + training_eps + episode,
        )

        pbar.update()
    pbar.close()
    
    # Finalize the experiment
    env.plot_results()

    plot_files = ["travel_times.png", "rewards.png"]
    images_to_log = {}
    for plot_file in plot_files:
        plot_path = os.path.join(plots_folder, plot_file)
        if os.path.exists(plot_path):
            images_to_log[f"plots/{plot_file.replace('.png', '')}"] = wandb.Image(plot_path)
    if images_to_log:
        wandb.log(images_to_log)

    loss_records = [
        {"iteration": iteration, "model": "T_coop", "loss": loss_value}
        for iteration, loss_value in enumerate(coop_model.loss_actor, start=1)
    ] + [
        {"iteration": iteration, "model": "cluster0_selfish", "loss": loss_value}
        for iteration, loss_value in enumerate(selfish_model.loss_actor, start=1)
    ]
            
    save_loss_records(
        records_folder,
        loss_records,
        columns=["iteration", "model", "loss"],
    )

    diagnostics_folder = os.path.join(records_folder, "losses")
    os.makedirs(diagnostics_folder, exist_ok=True)
    coop_diagnostics = pd.DataFrame(coop_model.diagnostics)
    coop_diagnostics.insert(0, "model", "T_coop")
    selfish_diagnostics = pd.DataFrame(selfish_model.diagnostics)
    selfish_diagnostics.insert(0, "model", "cluster0_selfish")
    combined_diagnostics = pd.concat([coop_diagnostics, selfish_diagnostics], ignore_index=True)
    combined_diagnostics.to_csv(
        os.path.join(diagnostics_folder, "subav_ifppo_diagnostics.csv"),
        index=False,
    )

    env.stop_simulation()
    clear_SUMO_files(os.path.join(records_folder, "SUMO_output"), os.path.join(records_folder, "episodes"), remove_additional_files=True)
    run_metrics_analysis(exp_id, results_folder=results_root)
    run_cluster_metrics(exp_id, results_folder=results_root)
    wandb.finish()

if __name__ == "__main__":
    main()

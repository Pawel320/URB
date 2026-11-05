## (MA)RL algorithms and baselines.

### (MA)RL algorithms

We deliver here scripts for the experiment runs. Each associated algorithm with selected implementations from `TorchRL`:
* ```ippo_torchrl.py``` uses Independent Proximal Policy Optimization algorithm,
* ```mappo_torchrl.py``` uses Multi Agent Proximal Policy Optimization algorithm,
* ```iql_torchrl.py``` uses Implicit Q-Learning algorithm,
* ```qmix_torchrl.py``` uses QMIX algorithm,
* ```vdn_torchrl.py``` uses Value Decomposition Network algorithm.

Moreover, we have two independent algorithms with our custom implementations:
* ```iql.py``` uses Independent Q-Learning,
* ```ippo.py``` uses Independent Proximal Policy Optimization,
* ```hyp_ippo.py``` uses Independent Proximal Policy Optimization with hypernetworks.

You can tune, adjust, hyperparameterize and modify all the provided implementations, or create own scripts.

### SubAV experiments

```subav_ifppo.py``` splits the AV fleet into two clusters and trains them with separate
IF-PPO policies in a single shared simulation. The split is given by an external grouping
CSV (columns ```agent_id``` and ```cluster``` (or ```cluster_id```), labels 0/1):

* cluster 1 = "T" set (AVs labeled 1): collaborative reward ```-0.5 * own_TT - 0.5 * mean_TT_of_T```,
* cluster 0 (AVs labeled 0): selfish reward ```-1 * own_TT```,
* humans labeled 1 are **not** part of T (T = AV ∧ cluster 1).

Reward coefficients come from the environment (RouteRL behaviors stamped per cluster after
mutation); the collaborative "group" term is scoped to T members only via a runtime patch of
```MachineAgent.get_reward```. There is no script-side reward pooling.

Usage (from ```scripts/```):

```
python subav_ifppo.py --id subav_run_1 --alg-conf config1 --task-conf subav_40 \
    --net saint_arnoult --grouping-file path/to/grouping.csv
```

* ```--grouping-file``` is optional if ```grouping_file``` is set in a merged config; a relative
  path is resolved from the launch directory. All machine agents must appear in the CSV and both
  clusters must be non-empty among the AVs.

### Output conventions

- At the end of each experiment script, metrics are automatically computed by calling `analysis/metrics.py`.
- For learning-based scripts, training losses are saved in a unified CSV file:
  `results/<exp_id>/losses/losses.csv`

### Baselines

In addition to RL algorithms, we provide baseline algorithms for comparison.
They can be executed with ```scripts/baselines.py``` or as standalone scripts ```scripts/<model_name>.py```, depending on the selected model.

The options consist of:

| Method   |  Description                                            | Location          | Execution     | Source          |
| -------- | ------------------------------------------------------- | ----------------- | ------------- | --------------- |
| `greedy` | Selects the route with the lowest recorded travel time based on past episodes. Uses a global structure to store per-agent past records. | `scripts/`     | Run as standalone script: `scripts/greedy.py` | Included in URB 
| `aon`    | Deterministically picks the shortest free-flow route regardless of congestion. | `baseline_models/` | Run via `scripts/baselines.py` with `--model aon` | Included in URB
| `random` | Fully undeterministic. | `baseline_models/` |     Run via `scripts/baselines.py` with  `--model random` | Included in URB |
| `gawron` | Human learning model based on [Gawron (1998)](https://kups.ub.uni-koeln.de/9257/); iteratively shifts cost expectations toward received rewards. | `baseline_models/` | Run via `scripts/baselines.py` with `--model gawron` | [RouteRL](https://github.com/COeXISTENCE-PROJECT/RouteRL/blob/993423d101f39ea67a1f7373e6856af95a0602d4/routerl/human_learning/learning_model.py#L42) |


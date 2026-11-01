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
```MachineAgent.get_reward``` (RouteRL-URB itself is unmodified). There is no script-side
reward pooling.

Usage (from ```scripts/```):

```
python subav_ifppo.py --id subav_run_1 --alg-conf config1 --task-conf subav_40 \
    --net saint_arnoult --grouping-file path/to/grouping.csv
```

* ```--grouping-file``` is optional if ```grouping_file``` is set in a merged config; a relative
  path is resolved from the launch directory. All machine agents must appear in the CSV and both
  clusters must be non-empty among the AVs.
* ```--results-dir <root>``` redirects all outputs (episodes, metrics, plots) to
  ```<root>/<id>/``` (default: ```URB/results/<id>/```). Used by the Slurm scripts to write
  to node-local scratch.
* Per-cluster behaviors are configured with ```cluster_behaviors``` (e.g. in the task config):
  ```{"0": "selfish", "1": "collaborative"}```.
* Default env config: ```--env-conf clusters-eta-mask``` (ETA masked-normalized observations).
* Outputs besides the usual ones: ```results/<id>/agent_clusters.csv``` (grouping used),
  ```results/<id>/metrics/ClusterMetrics.csv``` + ```metrics/plots/tt per cluster.png```
  (mirrored to ```plots/tt per cluster.png``` beside the environment plots),
  ```results/<id>/losses/subav_ifppo_diagnostics.csv```.
* Cluster metrics can be recomputed standalone: ```python analysis/cluster_metrics.py --id <id>```
  (or via ```scripts/utils.py:run_cluster_metrics```).
* Smoke test: ```pytest tests/test_subav_ifppo.py``` (SUMO required).
* For validation runs use a generated round-robin grouping (real groupings come from your own
  computation): ```python scripts/make_grouping_roundrobin.py --net ingolstadt_custom2
  --out config/groupings/ingolstadt_custom2_roundrobin.csv```.
* For a random baseline, shuffle agents into clusters 0/1 deterministically by --seed:
  ```python scripts/make_grouping_random.py --net ingolstadt_custom2 --seed 42
  --out config/groupings/ingolstadt_custom2_random_seed42.csv --ratio 0.5```.

### Server runs (Slurm, Singularity SUMO)

Slurm wrappers live in ```scripts/subav_server_scripts/``` and follow the ASGN conventions
(partition ```rknodes```, QoS ```big_bonk```, SUMO via ```sumo1_26_0patched.sif```, shared
container venv). They require **both** repos on the server:
```/home/$USER/URB``` and ```/home/$USER/RouteRL-URB``` (override with ```PATH_PROGRAM``` /
```PATH_ROUTE```).

Smoke the pipeline first (2 tasks x ~20 min on ingolstadt_custom2):

```bash
cd URB/scripts/subav_server_scripts
GROUPING_FILE=/home/$USER/URB/config/groupings/ingolstadt_custom2_roundrobin.csv \
ALG_CONF=test TASK_CONF=test NUM_RUNS=2 TIME_LIMIT=1:00:00 RUN_TAG=smoke \
./run_subav_full_pipeline.sh
```

Full seed sweep (default net ingolstadt_custom2, config1 x subav_40, 5 seeds):

```bash
GROUPING_FILE=/home/$USER/URB/config/groupings/my_split.csv \
NUM_RUNS=5 SEED_BASE=42 TIME_LIMIT=48:00:00 \
./run_subav_full_pipeline.sh
```

Notes:
* Each array task is one experiment: ```ENV_SEED = SEED_BASE + task_index``` (torch seed analog).
* Results go to node-local ```/scratch/tmp/$USER/subav``` — **rsync them off after the job**.
* Runtime on ingolstadt_custom2: ~38 s/episode (1035 agents); the full ```subav_40``` recipe
  (300 + 4000 + 100 episodes) takes ~46 h. Reduce ```training_eps``` in the algo config or lower
  ```human_learning_episodes```/```test_eps``` in the task config for shorter runs.
* Task logs: ```scripts/subav_server_scripts/container_printouts/output_<jobid>_<task>.txt```.

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


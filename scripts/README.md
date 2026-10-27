# Benchmark scripts

RoboDojo and MiVerse use one CLI and shared evaluation scripts. Run commands
from the repository root and select the benchmark with `--bench`:

```bash
bash scripts/evaluate.sh <command> --bench RoboDojo [options]
bash scripts/evaluate.sh <command> --bench MiVerse [options]
```

RoboDojo is the default. `--dataset` remains an alias for `--bench`.
`bash scripts/MiVerse/miverse.sh <command>` remains a thin compatibility alias
for MiVerse. For MiVerse assets, profiles, and scoring,
see the [MiVerse guide](MiVerse/README.md).

## Public entry points

| Script | Purpose |
| --- | --- |
| [evaluate.sh](evaluate.sh) | Shared CLI: `doctor`, `eval`, `server`, `client`, `smoke`, `benchmark`, `dimensions`, `summarize`, `tasks` |
| [install.sh](install.sh) | One-time environment setup (conda, Isaac Sim, submodules) |
| [init_assets.sh](init_assets.sh) | Download assets for `--bench RoboDojo`, `MiVerse`, or `all` (default: `all`) |
| [MiVerse/download_data.sh](MiVerse/download_data.sh) | Download `ridiculouslzx/MiVerse` from ModelScope using a token, parallel files and large-file range threads; [usage](MiVerse/README.md#download-the-modelscope-dataset) |
| [eval_policy.sh](eval_policy.sh) | Isaac Sim eval client (called by `evaluate.sh client` and XPolicyLab) |

Benchmark metadata selects the task directory, default environment profile,
result directories, and summary implementation. RoboDojo defaults to `arx_x5`;
MiVerse uses the `miverse` profile with `config_name: miverse` for layouts and
results. Both use the `RoboDojo` simulator Conda environment. An explicit
`--env-cfg` must belong to the selected benchmark. The policy runs in its own
environment: `--policy-env` does not install dependencies or activate the
simulator for standalone `client` commands.

| Command | RoboDojo | MiVerse |
| --- | --- | --- |
| `tasks`, `doctor` | Task inventory and preflight checks | Task inventory and preflight checks |
| `eval` | One task with a fresh local policy server | Same |
| `server`, `client` | Split evaluation; omit `--task` for batch workers | Same |
| `smoke`, `benchmark` | Task filters, GPU grouping, logs, resume | Same |
| `summarize` | Original RoboDojo scoring | MiVerse's 24-task, three-seed scoring |
| `dimensions` / `--dimension` | Five RoboDojo capability dimensions | Not applicable; use task filters |

## Download assets

Run the asset downloader after installing the simulator environment:

```bash
bash scripts/init_assets.sh                  # both benchmarks (default: all)
bash scripts/init_assets.sh --bench RoboDojo # RoboDojo only
bash scripts/init_assets.sh --bench MiVerse  # MiVerse only
bash scripts/init_assets.sh --help
```

Unlike `evaluate.sh`, the asset downloader defaults to **both** benchmarks.
It selects the benchmark's object, layout, robot, and room packages, plus
shared `Material/` and `Background/` assets. Shared `Sensor/` and `Traj/`
packages are included when present in the source. Downloading another
benchmark preserves previously installed assets and local files not supplied
by the selected revision.

Both benchmarks use the same source, with these environment overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HF_REPO_ID` | `RoboDojo-Benchmark/RoboDojo` | Hugging Face dataset repository |
| `HF_REPO_URL` | `https://huggingface.co/datasets/${HF_REPO_ID}` | Explicit Git repository URL; overrides the URL derived from the ID |
| `HF_REVISION` | `main` | Asset revision to download |

The source stores packages under `Assets/`. The selected revision must contain
all packages required by the requested benchmark; the downloader fails with a
missing-package error otherwise. In particular, use a revision containing
MiVerse assets for `--bench MiVerse` or `--bench all`.

Completion is checked against the source file list and sizes, and unresolved
Git LFS pointers are rejected. Empty directories do not count as downloaded
assets. This verifies the download; use `evaluate.sh doctor` to check a task's
native layout count and environment, then run a smoke test to verify simulation.

## Typical eval flow

```text
evaluate.sh eval --bench <NAME>
  -> scripts/internal/run_policy_eval.sh
    -> policy server (localhost) + sim client

Split / multi-machine:

evaluate.sh server --bench <NAME>  ->  scripts/internal/run_policy_server.sh  ->  policy server
evaluate.sh client --bench <NAME>  ->  scripts/eval_policy.sh  ->  src/eval_client/main.py
```

Use `bash scripts/evaluate.sh server --help` and
`bash scripts/evaluate.sh client --help` for split-evaluation options.
The [MiVerse split example](MiVerse/README.md#split-server-and-client) shows
matching benchmark, policy, profile, checkpoint label, action type, seed, and
port on both sides. `client` runs in the active simulator environment and does
not load or change the server's checkpoint; its `--ckpt` / `--action-type`
values label the results and must match the server.

Inspect tasks or validate an installation for either benchmark:

```bash
bash scripts/evaluate.sh tasks --bench RoboDojo --check
bash scripts/evaluate.sh tasks --bench MiVerse --check
bash scripts/evaluate.sh doctor --bench MiVerse \
  --task place_bread_in_oven --seed 0 --eval-num native --skip-policy
```

`doctor` checks the chosen task and seed, including the layout count needed for
the requested evaluation. Asset checks follow task-specific robot/scene
overrides and saved layout references: robot USD, URDF and required planner
configs, object USD and metadata, room USD, and configured background/material
files must exist, be nonempty, and contain downloaded data rather than Git LFS
pointers. Unused assets from the other benchmark are not required. USD-internal
dependencies and actual asset loading still need a smoke test.

Checkpoint checks use XPolicyLab's runtime resolver. `--ckpt` accepts a short
name, full training-run directory name, absolute path, or path relative to the
policy directory; adapter-specific deploy/environment overrides take their
runtime precedence. Match `--action-type` and `--seed` to the evaluation
(`--action-type` defaults to `ee`). For example:

```bash
bash scripts/evaluate.sh doctor --bench MiVerse \
  --task place_bread_in_oven --policy-dir XPolicyLab/policy/Pi_05 \
  --ckpt /path/to/checkpoint --action-type joint --seed 0 --eval-num 1
```

This checks the checkpoint path, without loading weights or validating model
artifacts. The weight-free `demo_policy` does not require a checkpoint.
`doctor` does not launch a simulation. The `--skip-isaac`,
`--skip-conda`, and `--skip-policy` flags intentionally omit those checks; they
do not make an asset-free checkout ready to evaluate.

Run one or more official RoboDojo capability dimensions in a benchmark sweep:

```bash
bash scripts/evaluate.sh dimensions --bench RoboDojo
bash scripts/evaluate.sh benchmark --bench RoboDojo \
  --dimension memory,long-horizon \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --ckpt <CHECKPOINT> \
  --policy-env <ENV> \
  --eval-num native
```

Available RoboDojo dimensions are `generalization`, `memory`, `precision`,
`long-horizon`, and `open`. Generalization includes both the 12 standard tasks
and their 12 runnable `_random` layout variants. Combine `--dimension` with
`--only` or `--tasks-file` to narrow a dimension further. MiVerse has 24 tasks
(eight base tasks and their `_l`/`_s` variants) and no capability dimensions;
select its tasks with `--only`, `--tasks-file`, or `--limit`.

## Batch execution and multi-GPU grouping

`evaluate.sh smoke` and `evaluate.sh benchmark` use the same scheduler for both
benchmarks. Execution is sequential by default. `smoke` defaults to one episode
per task; `benchmark --eval-num native` uses each task's configured episode
count. Pass `--gpu-ids` to run task groups in parallel, with tasks inside each
group running sequentially. Separate `--policy-gpu-ids` and `--env-gpu-ids` lists
can assign policy servers and simulator clients to different devices.

Grouping depends on the selected benchmark and environment profile:

- **RoboDojo / `arx_x5`:** retains the original measured runtime weights and
  grouping behavior. The weights are stored in `internal/benchmark_groups.py`.
- **MiVerse:** has no measured runtime weights. Tasks are assigned round robin
  by task count; equal task counts do not guarantee equal execution time.
- Profiles without runtime weights also use round-robin task assignment.

The examples use the included [demo policy](../XPolicyLab/policy/demo_policy/README.md),
which holds the observed pose and requires no model checkpoint. Prepare its
`demo-policy` environment separately from the `RoboDojo` simulator environment
before an actual run. Batch scripts require Bash 4 or newer.

From the repository root, prepare the demo environment once:

```bash
conda create -n demo-policy python=3.11 -y
conda activate demo-policy
bash XPolicyLab/policy/demo_policy/install.sh
conda activate RoboDojo
```

Real policies need their own installation and compatible weights. Choose
`--action-type` explicitly: the CLI defaults to `ee`, while several included
MiVerse adapters require `joint`. See the
[compatibility list](MiVerse/README.md#policy-and-action-compatibility).

Preview a RoboDojo sweep:

```bash
bash scripts/evaluate.sh benchmark --bench RoboDojo \
  --policy-dir XPolicyLab/policy/demo_policy \
  --ckpt demo \
  --policy-env demo-policy \
  --action-type joint \
  --eval-num 1 \
  --gpu-ids 0,2,5,7 \
  --dry-run
```

Preview all 24 MiVerse tasks split across four GPUs:

```bash
bash scripts/evaluate.sh benchmark --bench MiVerse \
  --policy-dir XPolicyLab/policy/demo_policy \
  --ckpt demo \
  --policy-env demo-policy \
  --action-type joint \
  --eval-num native \
  --gpu-ids 0,1,2,3 \
  --dry-run
```

Remove `--dry-run` to launch. A batch task passes validation only when its process
exits successfully and `_result.json` has a valid integer `eval_time` covering
the requested episode count, capped by that task's native count. For example,
MiVerse `--eval-num 1` requires one completed episode, while `--eval-num native`
requires 25 for every task, as configured in
`task/MiVerse/config/_task.yml`. A partial native evaluation is marked `FAIL`
and is not skipped by resume. `PASS` describes evaluation completion, not policy
task success.
Task failures count as completed episodes; skipped or unstable layouts do not.
Use `smoke` or `benchmark` for this result-based completion check; a single-task
`eval` or `client` exit code alone does not establish that all requested episodes
were evaluated.

`--gpu-ids` distributes tasks across processes. In contrast, `EVAL_NUM_ENVS`
controls simultaneous environments inside each simulator process:

```bash
EVAL_NUM_ENVS=2 bash scripts/evaluate.sh eval --bench MiVerse \
  --policy-dir XPolicyLab/policy/Pi_05 --policy-env <POLICY_ENV> \
  --task place_bread_in_oven --ckpt <CHECKPOINT> \
  --action-type joint --eval-num 2
```

MiVerse defaults to one environment. Parallel environments require an adapter
with `eval_batch: true` and sufficient runtime resources. `demo_policy` declares
`eval_batch: false`, so the client forces one environment even when overridden.
The six learned adapters currently declare `eval_batch: true`; this is interface
support, not a guarantee that a given GPU/checkpoint can run every batch size.

### Logs and resume

Batch logs and status files default to `smoke_results/` for RoboDojo and
`smoke_results/MiVerse/` for MiVerse.
Set `--run-id NAME` for a repeatable sweep identity, then rerun with `--resume`
to skip completed tasks. Keep the same execution mode (`eval` or `client`),
benchmark, profile, policy, checkpoint, action type, seed, and episode count.
Alternatively, point `--summary PATH` at the original summary. Server mode
starts persistent servers and does not support `--resume`.

On Ctrl-C or SIGTERM, parallel sweeps save finished workers' results before
exiting. Resume skips those completed tasks and reruns unfinished ones.

For independent comparisons, restart externally managed policy servers when
their sampling RNG must start from the same state; an episode-level reset does
not necessarily reset the model's RNG.

## Result summaries

```bash
bash scripts/evaluate.sh summarize --bench RoboDojo
bash scripts/evaluate.sh summarize --bench MiVerse
```

The shared command selects the summary script in the matching benchmark folder.
RoboDojo and MiVerse keep their own scoring and completeness rules:

| Benchmark | Summary script | Default output |
| --- | --- | --- |
| RoboDojo | [RoboDojo/summarize_result.py](RoboDojo/summarize_result.py) | `eval_result/RoboDojo/_summary.md` |
| MiVerse | [MiVerse/summarize_result.py](MiVerse/summarize_result.py) | `eval_result/MiVerse/_summary.md` |

When an adapter accepts a checkpoint path, pass the original path to `--ckpt`.
Result directory labels containing path separators are percent-encoded with a
`urlencoded:` prefix so the experiment remains in one directory. Model loading
still receives the original checkpoint argument; MiVerse summary filters also
accept that original value, including its full path.
Encoded labels exceeding 200 bytes are rejected; use a shorter checkpoint name
or path. This keeps result labels reversible without adding metadata to `_result.json`.

See the
[MiVerse guide](MiVerse/README.md#result-summaries) for its three-seed aggregation
and result filters.

RoboDojo selects the most recently modified `_result.json` for each
task/policy/result configuration/seed. Ties use the run-directory name, then
the outer run label. This supports custom `--run-id` values. Insufficient
episodes or invalid JSON in the latest result leave the cell incomplete;
the summary does not fall back to an older result.

For MiVerse, `--env-cfg` on `summarize` filters the saved `config_name`;
launch commands use it to select the profile stem. Both values are `miverse`
in the default `env_cfg/miverse.yml` profile.
MiVerse summaries select the run containing the most recently modified
`_result.json` for each task/policy/config/seed, breaking modification-time ties
by run-directory name. Within a config, an incomplete or invalid latest result
does not fall back to an older result. Configs are then combined into one policy
report: the last complete config in sorted order wins for each task/seed.
Use the checkpoint, action-type, and config filters for independent comparisons.
Reports separate clean, language (`_l`), and spatial (`_s`) dimensions, each
covering eight tasks and seeds `0,1,2` (24 cells per dimension, 72 overall).
Every cell requires 25 valid episode records. Three-seed averages require all
three seeds; incomplete task coverage is marked with a dagger.

## CPU validation

With Python 3.11+, PyYAML, NumPy, and Bash 4+:

```bash
bash scripts/evaluate.sh tasks --bench RoboDojo --check
bash scripts/evaluate.sh tasks --bench MiVerse --check
bash -n scripts/evaluate.sh scripts/eval_policy.sh
```

Task inventory checks, shell syntax checks, and `--dry-run` do not validate
loaded assets, rendering, physics, Pink IK, or learned checkpoint inference.
Run `doctor` and an actual `smoke` on the target machine before a full
benchmark; keep smoke completion separate from model success metrics.

## Internal (`internal/`)

These scripts are called by `evaluate.sh` or policy utilities.

| File | Purpose |
| --- | --- |
| [benchmark_profile.py](internal/benchmark_profile.py) | Benchmark metadata, environment defaults, and result configuration |
| [task_inventory.py](internal/task_inventory.py) | Shared task discovery; RoboDojo capability dimensions |
| [verify_install.sh](internal/verify_install.sh) | `evaluate.sh doctor` for the selected benchmark |
| [check_assets.py](internal/check_assets.py) | Static checks of selected robot, scene, object, and layout asset files |
| [check_policy_checkpoint.py](internal/check_policy_checkpoint.py) | Checkpoint path preflight using XPolicyLab's runtime resolver |
| [smoke_all_tasks.sh](internal/smoke_all_tasks.sh) | Shared `smoke` / `benchmark` runner |
| [benchmark_groups.py](internal/benchmark_groups.py) | Shared task grouping and RoboDojo runtime weights |
| [stat_score_distribution.py](internal/stat_score_distribution.py) | Offline score histogram analysis (manual) |

New benchmarks should register their metadata and supply their own tasks and
scoring rules, reusing this discovery, validation, launch, and batch path.

## Docker

Container install and smoke tests live under [../docker/](../docker/).

## Policy-specific scripts

Training, data prep, and per-policy `eval.sh` live in [../XPolicyLab/policy/](../XPolicyLab/policy/) (submodule).

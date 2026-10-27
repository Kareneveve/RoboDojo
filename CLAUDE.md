# RoboDojo — Claude Code Guide

## Project Overview

RoboDojo is a unified sim-and-real robotics benchmark built on NVIDIA Isaac Sim / IsaacLab. This fork adds **MiVerse as a separate benchmark** on that foundation. Policy evaluation runs via XPolicyLab and the shared simulation stack.

Project boundary: preserve the original RoboDojo tasks, task configs, reward/success logic, and scoring semantics. Do not refactor or modify them while adding or maintaining MiVerse. Use the shared `scripts/evaluate.sh` CLI with `--bench RoboDojo|MiVerse` for task selection, validation, launches, and batch evaluation. Keep benchmark-specific metadata and scoring separate; shared-stack changes must remain compatible with RoboDojo.

Contributors maintain **Tasks** (scene + reward logic), shared **env** / **env_cfg** configs, and eval infrastructure under **scripts/** and **src/eval_client/**.

Human-facing docs: [MiVerse](scripts/MiVerse/README.md) and [original RoboDojo](https://robodojo-benchmark.com/doc/).

## Repository Layout

```
task/RoboDojo/
  config/<task_name>.yml     task scene/object config
  tasks/<task_name>.py       task logic (inherits TaskEnv)
  demos/                     non-inventory demos (e.g. dlc)
task/MiVerse/                added mobile manipulation task logic and configs
task/task_registry.py        shared benchmark-scoped dynamic task loader
scripts/RoboDojo/            RoboDojo scoring and download helpers
scripts/MiVerse/             MiVerse scoring, guide, compatibility CLI alias
env/                         TaskEnv backbone and managers
env_cfg/                     robot / scene / sim / camera YAML
src/eval_client/             eval client (main.py, eval_env.py)
utils/                       paths, save/load, pipeline helpers
scripts/                     evaluate.sh, install.sh, init_assets.sh, internal/
docker/                      container eval (see docker/README.md)
XPolicyLab/                  policy server + deploy (submodule)
third_party/                 IsaacLab, curobo (submodules)
Assets/                      downloaded runtime assets for the selected benchmarks (not in git)
eval_result/                 runtime eval output
```

## Submodules & Ownership

| Own in RoboDojo | Own in XPolicyLab |
| :-- | :-- |
| `env/`, `env_cfg/`, `task/RoboDojo/`, `task/MiVerse/`, `src/eval_client/` | Policy code, checkpoints, `deploy.yml`, policy server |
| `scripts/evaluate.sh`, `scripts/eval_policy.sh` | `XPolicyLab/policy/<POLICY>/eval.sh`, `setup_eval_*` scripts |
| `utils/`, install/assets scripts | Policy-specific dependencies and training |

- Submodule URL: `https://github.com/XPolicyLab/XPolicyLab.git` (branch `main`; see `.gitmodules`).
- Update the gitlink pin intentionally; do not edit submodule contents unless updating the pin.
- RoboDojo eval client uses `XPolicyLab/client_server/ws/model_client.py` (WebSocket transport).

## Eval Commands & Flow

**CLI** (`bash scripts/evaluate.sh <command> --bench RoboDojo|MiVerse`):

The default benchmark is RoboDojo; `--dataset` is a compatibility alias for
`--bench`. `scripts/MiVerse/miverse.sh` is a thin compatibility entry point
for MiVerse. Add benchmark metadata to the shared infrastructure instead of creating
a separate task inventory, doctor, or batch runner for each benchmark. Defaults
and summary dispatch live in `scripts/internal/benchmark_profile.py`; grouping
and the existing RoboDojo runtime weights live in `scripts/internal/benchmark_groups.py`.

| Command | Purpose |
| :-- | :-- |
| `doctor` | Check assets, configs, env before eval |
| `eval` | One task: policy server + sim client on localhost |
| `server` | Policy server only (split / multi-machine eval) |
| `client` | Sim client only (connect to running server) |
| `smoke` | Selected/all tasks, `EVAL_NUM=1` by default |
| `benchmark` | Full sweep (`--eval-num native` or numeric); sequential unless GPU lists are supplied |
| `dimensions` | List RoboDojo capability dimensions; MiVerse task selection accepts only `all` |
| `summarize` | Aggregate `eval_result/` into markdown table |
| `tasks` | List runnable tasks (maintainer inventory) |

**Single-machine eval:**

```
evaluate.sh eval
  → scripts/internal/run_policy_eval.sh
    → (cd policy_dir) setup_eval_policy_server.sh   # policy server
    → setup_eval_env_client.sh                    # sim client → src/eval_client/main.py
```

- Policy `eval.sh` must exist under `--policy-dir` (used for validation and optional `expert_num`); `evaluate.sh eval` does **not** invoke `eval.sh` directly.
- Policy `setup_eval_*` scripts run with **CWD = policy directory** (relative paths in policies).
- Default transport: **WebSocket** (`protocol: ws` in policy `deploy.yml`; see `scripts/eval_policy.sh`).
- Explicitly choose the action contract. MiVerse FastWAM/Pi_05/MM_ABC/MoPA use `joint`; OpenWAM/Xiaomi_Robotics_1 use `ee`; demo supports either. Checkpoint layouts are adapter-specific even when their vectors have the same width.
- Policy installation is separate from simulator installation. The demo environment can be prepared with `conda create -n demo-policy python=3.11`, activation, then `bash XPolicyLab/policy/demo_policy/install.sh`.

**Split eval:**

```
evaluate.sh server  →  scripts/internal/run_policy_server.sh
evaluate.sh client  →  scripts/eval_policy.sh  →  src/eval_client/main.py
```

**Docker:** sim client in container, policy server on host — see `docker/README.md`.

Fast validation:

```bash
bash scripts/evaluate.sh doctor --skip-isaac --skip-conda --skip-policy
bash scripts/evaluate.sh eval --policy-dir XPolicyLab/policy/<POLICY> --task stack_bowls --ckpt <CKPT> --policy-env <ENV> --dry-run
bash scripts/evaluate.sh smoke --policy-dir XPolicyLab/policy/<POLICY> --ckpt <CKPT> --policy-env <ENV> --only stack_bowls,push_T --dry-run
```

For a sequential eval-infrastructure acceptance run:

```bash
bash scripts/evaluate.sh smoke --policy-dir XPolicyLab/policy/<POLICY> --ckpt <CKPT> --policy-env <ENV> --fail-fast
```

Multi-GPU batching uses the shared scheduler. Preserve RoboDojo's original
`arx_x5` runtime weights and capability dimensions. MiVerse has no runtime
weights and uses round-robin task assignment; do not describe it as balancing
execution time. Result summaries dispatch to each benchmark's existing scoring
implementation without changing score semantics: `scripts/RoboDojo/summarize_result.py`
and `scripts/MiVerse/summarize_result.py`. Keep these implementations in their
respective benchmark folders. The default output is
`eval_result/<benchmark>/_summary.md`.

New code imports from `env.*`.

## Tasks

MiVerse adds **24 tasks** under `task/MiVerse/`: eight base tasks and their `_l`
(language) and `_s` (spatial) variants. Use
`bash scripts/evaluate.sh tasks --bench MiVerse --check` to inspect them. Its default
profile is `miverse`, with layout/result `config_name: miverse`.
The original RoboDojo inventory below remains unchanged:

- **54 runnable task configs** (42 base + 12 `_random` generalization variants).
- Registration is dynamic: `task_registry.load_task_class(task_name)` imports `task.RoboDojo.tasks.<task_name>` and expects class name == module name.
- Config path: `task/RoboDojo/config/<task_name>.yml`.
- Inventory: `bash scripts/evaluate.sh tasks` or `python scripts/internal/task_inventory.py --format json --check`.
- `task/RoboDojo/demos/` (e.g. `dlc`) is outside the benchmark inventory scan.

## Naming

- **Config / scripts**: lowercase `snake_case` — `env_cfg/`, `utils/`, `scripts/`
- **Python packages/directories**: `snake_case` — `env/robot_manager/`, `env/scene_manager/objects/`
- **Python module files**: `snake_case.py` — `task_env.py`, `obs_manager.py`
- **Python classes**: **PascalCase** — `RewardManager`, `TaskEnv`
- **Tasks**: filename, YAML name, exported env class, and layout/result paths must match. Most tasks use lowercase `snake_case`; **`play_Xylophone`**, **`swap_T`**, **`push_T`**, and **`push_T_random`** match uppercase asset names.
- **Submodules**: keep upstream casing — `XPolicyLab/`, `Assets/`

## Code Conventions

### Task File Structure

Every task Python file must follow this pattern:

```python
class <TaskName>Common:
    def __init__(self, config, app, **kwargs): ...
    def _post_setup_scene(self, sim): ...
    def reset(self, seed=None, options=None): ...
    def soft_reset(self, seed=None, options=None): ...  # when state must reset between episodes
    def run_reward(self): ...                           # reward / success check

class <task_name>(<TaskName>Common, TaskEnv):
    pass
```

- Class names use **PascalCase**; the exported env class uses **snake_case** matching the YAML / Python module basename.
- `run_reward()` must call `self.reward_manager.check(...)` (or equivalent) meaningfully — never leave success checks trivially always `True`.
- No print statements in task logic; use the shared logger if needed.

### Config File (YAML)

Typical rigid-body tasks define:
- `Rigid` section with at least one object category and a `label` list.
- `common` with `xlim`, `ylim`, `rotate_rand` (and `rotate_deg` when rotation is enabled).
- `select_mode` with `nums` and `mode`.

Garment tasks (e.g. `fold_clothes*`) use a `Garment` section instead of `Rigid`. Many configs add `Geometry`, `Clutter`, etc.

### Commit Message Format

```
[Scope] type: short description

Scopes: Task | Env | Config | scripts | utils | fix | refactor
Types : feat | fix | update | refactor | docs | chore
```

Examples:
- `[Task] update: fix stack_bowls run_reward bowl_color check`
- `[Env] fix: correct gripper warm-up timing`
- `[Config] update: adjust xlim for insert_test_tubes`

### Branch Naming

`<contributor>/<scope>-<short-description>`  e.g. `zixuan/task-hang-mugs`

## Validation Loops

Use the **smallest loop** that proves the current change.

### Docs / scripts / config only

```bash
bash -n scripts/evaluate.sh scripts/eval_policy.sh
python scripts/internal/task_inventory.py --bench RoboDojo --format json --check
python scripts/internal/task_inventory.py --bench MiVerse --format json --check
ruff check .
git diff --check
```

These checks validate syntax and task inventories without launching Isaac Sim
or loading checkpoints. Run additional focused tests when available. Do not
describe dry runs or CPU checks as proof of GPU simulation, asset loading,
Pink IK, or actual model inference.

Asset checks require real assets. `scripts/init_assets.sh --bench
RoboDojo|MiVerse|all` selects packages; its default is `all`, while the evaluation
CLI defaults to RoboDojo. Both benchmarks use the same Hugging Face asset
source, configurable through `HF_REPO_ID`, `HF_REPO_URL`, and `HF_REVISION`.
The selected revision must contain the requested benchmark's assets; missing
packages must fail rather than appear ready. MiVerse needs complete
`Assets/Object/MiVerse/`, `Assets/Room/MiVerse/`, `Assets/Robots/m92uw/`, and
`Assets/Eval_Layout/MiVerse/miverse/<seed>/` packages, plus shared materials and
backgrounds. Download verification checks file lists, sizes, and unresolved
Git LFS pointers; it does not prove GPU simulation or native layout coverage.
Check each requested task/seed/count with `evaluate.sh doctor --bench BENCH
--task TASK --seed N --eval-num NUM|native`; omit skip flags for a full preflight.
Doctor asset checks follow task-specific scene/robot overrides and saved layout
references, rejecting missing/empty files and unresolved LFS pointers. Keep
these checks offline; USD dependency loading still requires a real smoke test.
Doctor checkpoint checks use the XPolicyLab runtime resolver and the same
`--action-type` (default `ee`) and seed as evaluation. Accept short names,
full run names and paths; checking path existence is not weight validation.

### Dry-run eval path (no Isaac, no policy server)

```bash
bash scripts/evaluate.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task stack_bowls \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --dry-run
bash scripts/evaluate.sh smoke \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --only stack_bowls,push_T \
  --dry-run
```

### Runtime (Isaac Sim + policy ready)

```bash
bash scripts/evaluate.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task stack_bowls \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --eval-num 1
```

Smoke/benchmark acceptance: each task must exit `0` **and** write `_result.json`
with a valid integer `eval_time >= min(requested, native)`. `native` uses the
task's declared count in `_task.yml`: 25 for every MiVerse task, including
`_l`/`_s` variants. Partial runs must fail completion checks and must not be skipped
on resume. This is run completion, not policy task success.
This validation belongs to the shell smoke/benchmark runner. A single-task
simulator exit code alone does not establish that the requested count was met.

Resume identities include benchmark, profile, policy, checkpoint, action type,
seed, episode count, and execution mode (`eval` or `client`). Persistent server
mode does not support resume. In split deployment, match these policy settings
and the port mapping across server/client; standalone clients run in the active
simulator environment and only label results with `--ckpt` / `--action-type`.

MiVerse defaults to one environment per simulator; `EVAL_NUM_ENVS` overrides
that count when the adapter has `eval_batch: true`. `eval_batch: false` forces
one environment. Multi-GPU groups are separate processes and must not be confused
with parallel environments within a process.

RoboDojo summary and its video checks select the same latest `_result.json`
per task/policy/result configuration/seed, using file modification time, then
run-directory name and outer run label for ties. Custom `--run-id` values must
not determine chronology. Select before validating JSON or episode count so
an incomplete latest result cannot fall back to an older complete run.

MiVerse summaries report clean, language (`_l`), and spatial (`_s`) dimensions,
each with eight tasks × seeds `0,1,2` (24 cells per dimension, 72 overall).
Every task/seed cell requires 25 valid episode records.
`summarize --env-cfg` filters result `config_name`; launch `--env-cfg` selects
the profile stem. Both values are `miverse` in the default profile. Select the
most recently modified `_result.json` per task/policy/config/seed, breaking ties
by run-directory name. Invalid or incomplete latest results must not silently
fall back to older runs within that config. Then combine configs in sorted
order into one report per policy: the last complete config wins for a task/seed.
Use checkpoint, action-type, and config filters for independent comparisons.
Three-seed averages require all three seeds; mark partial task coverage with a
dagger and preserve the upstream clean/language/spatial ranking rules.

## PR Checklist

### Task PR
- [ ] `config/<task>.yml` + `tasks/<task>.py` both present, names match
- [ ] `run_reward()` calls `self.reward_manager.check(...)` meaningfully
- [ ] YAML labels match Python labels exactly
- [ ] `step_lim` is appropriate for task length
- [ ] Task inherits `TaskEnv` (not `SyncCollectEnv` / `SyncRobotEnv`)
- [ ] New task filenames are lowercase `snake_case` (unless uppercase asset exception)

### Framework / Env PR
- [ ] Backward-compatible; no new mandatory constructor args
- [ ] `reset()` and `soft_reset()` both cover new state when applicable

### Config PR
- [ ] YAML valid; defaults safe; no silent breakage to shared configs

### General
- [ ] Commit messages: `[Scope] type: desc`
- [ ] No `print`, `breakpoint`, or commented-out debug code
- [ ] New files follow Naming rules above
- [ ] For scripts/eval infrastructure, run `evaluate.sh doctor` and at least dry-run `evaluate.sh smoke` for each affected `--bench` before requesting review

---

## Review Instructions for Claude

When asked to review a PR (`/review`):

1. **1-sentence summary** + PR type (Task / Framework / Config / Scripts / Mixed).
2. **Run the relevant checklist** from above. Skip inapplicable sections.
3. **Report only real issues** — skip items that pass. For each issue: file:line, what's wrong, one-line fix.
4. **Score**: ✅ Approve / ⚠️ Approve with comments / ❌ Request changes.

Type-specific red flags to check first:
- *Task*: YAML labels ≠ Python labels? Still using `SkillManager` / `SyncCollectEnv` / `pre_*`?
- *Framework*: breaks existing tasks or eval client? `reset`/`soft_reset` miss new state?
- *Config*: YAML valid? shared config silently breaks other tasks?

When creating a PR (`gh pr create`):
- Title: `[Scope] type: description (≤70 chars)`
- Body: summary bullets, affected tasks, test plan (`bash scripts/evaluate.sh eval ...` or task-specific eval).

---

## Common Pitfalls

- `check_success` always `True` → garbage eval metrics
- YAML label `["target"]` but code uses `label="obj"` → silent mismatch
- Missing `soft_reset` override → state leak between episodes
- `step_lim = 200` on long-horizon tasks → truncation artifacts
- Camera config changes affecting all tasks without announcement
- Treating a launcher exit code alone as sweep completion → also require `_result.json` to cover the effective requested episode count
- Treating empty asset directories or a download check as proof of native layout coverage and GPU simulation
- Mixing MiVerse `joint` / `ee` checkpoint layouts or confusing profile names with saved `config_name` values
- Editing policy logic in RoboDojo instead of XPolicyLab
- Running policy `setup_eval_*` without CWD in the policy directory
- Treating submodule content as owned by this repo unless explicitly updating the gitlink pin

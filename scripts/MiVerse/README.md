# MiVerse evaluation scripts

Use the independent MiVerse CLI from the repository root:

```bash
bash scripts/MiVerse/miverse.sh <command> [options]
```

Commands are `tasks`, `doctor`, `eval`, `server`, `client`, `smoke`, `benchmark`,
and `summarize`. The default benchmark is `MiVerse`, the default environment
profile is `miverse`, and the simulator uses the shared `RoboDojo` Conda
environment. No extra installation script or `--dataset` option is needed.

The single-task `eval`, `server`, and `client` commands delegate to the existing
`scripts/robodojo.sh` with explicit MiVerse settings. Batch execution and result
summarization are implemented in this directory; the original RoboDojo CLI and
its task selection and scoring rules remain unchanged.

## Inspect and run tasks

```bash
bash scripts/MiVerse/miverse.sh tasks --check
bash scripts/MiVerse/miverse.sh doctor --skip-isaac --skip-conda --skip-policy
bash scripts/MiVerse/miverse.sh eval \
  --task place_kettle_on_tv_stand \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --ckpt <CHECKPOINT> \
  --policy-env <POLICY_ENV> \
  --action-type joint \
  --seed 0 \
  --dry-run
```

Remove `--dry-run` to evaluate. Set `--action-type` to the interface required by
the policy; the default is `ee`. For split evaluation, use `server` and `client`
with the same policy settings and set `--policy-host` / `--policy-port` on the
client as needed.

The task inventory includes eight base tasks and their `_l` (language) and `_s`
(spatial) variants, for 24 tasks in total. Use the full task name, for example
`--task place_kettle_on_tv_stand_l`. Each variant uses its own eval layouts under
`Assets/Eval_Layout/MiVerse/miverse/<seed>/<task>_<layout_id>.json`.

For `_l` tasks, the instruction is the exact nonempty `instruction` saved in each
layout; missing text is an error. Instructions refresh when the next layout
batch loads. For `place_bread_in_oven_s`, the pickup surface is taken from that
layout's bread object (`kitchen_island`, `cutting_board`, or `table`), so an old
generic or mismatched instruction is rebuilt to name the actual surface.
`throw_away_trash_s` refreshes the package name for each layout batch;
`put_object_in_sink_s` retains the source task's generic instruction.

## Batch evaluation

```bash
bash scripts/MiVerse/miverse.sh benchmark \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --ckpt <CHECKPOINT> \
  --policy-env <POLICY_ENV> \
  --action-type joint \
  --seed 0 \
  --eval-num native \
  --dry-run
```

`smoke` defaults to one episode per task. `benchmark --eval-num native` uses the
episode counts in `task/MiVerse/config/_task.yml` (50 for base tasks and 25 for
each `_l`/`_s` variant), even if `EVAL_NUM` is set in the parent shell. Repeat the
command with `--seed 1` and `--seed 2` for the three-seed summary.

Select tasks with `--only task_a,task_b`, `--tasks-file PATH`, or `--limit`.
MiVerse has 24 tasks and no capability dimensions. Execution is sequential
by default. To divide tasks evenly across GPUs, add `--gpu-ids 0,1,2,3`.

In batch `eval` mode, each task starts a fresh policy server. Batch `client`
mode connects to external servers via `--policy-host` / `--policy-port`.
Restart external servers before independent comparisons when matching policy
sampling RNG state is required; a policy's episode reset may not reset its RNG.

Batch status summaries and logs default to `smoke_results/MiVerse/`. To resume
an interrupted sweep, reuse `--run-id` (or `--summary`) with `--resume` and the
same profile, policy, checkpoint, action type, seed, and episode count. A task
passes batch validation only if its process exits successfully and its saved
result contains `eval_time >= 1`. This status indicates a valid evaluation run;
it does not indicate policy task success.

## Result summaries

The environment profile is named `miverse`. Layouts and saved results use its
`config_name`, also `miverse`:

```text
eval_result/MiVerse/<task>/<policy>/miverse/<seed>_ckpt_name=...,action_type=.../<run_id>/_result.json
```

```bash
bash scripts/MiVerse/miverse.sh summarize
bash scripts/MiVerse/miverse.sh summarize \
  --ckpt <CHECKPOINT> --action-type joint --env-cfg miverse
```

The summary script can also run directly:

```bash
python scripts/MiVerse/summarize_result.py \
  --root eval_result/MiVerse --output /tmp/miverse_summary.md
```

`--root` selects the result root containing task directories; otherwise
`MIVERSE_EVAL_ROOT` or `eval_result/MiVerse` is used. `--output` overrides the
default `<root>/_summary.md`. For summary filtering, `--env-cfg` is the result
**config_name directory**, currently `miverse`. To summarize older runs stored
under `miverse_teleop`, use `--env-cfg miverse_teleop` instead.

Each result configuration and run label, including checkpoint and action type,
forms a separate experiment. Summaries report success rate and score for all
24 tasks and seeds 0, 1, 2, with progress out of 72 task/seed cells. A cell
requires the task's native episode count (50 for base tasks, 25 for variants),
and only the first such records are scored. The latest timestamp containing a
result is used; an incomplete or invalid latest result does not fall back to an
older completed result. Incomplete cells stay blank, and invalid files are
reported in the terminal.

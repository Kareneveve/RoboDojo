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
episode counts in `task/MiVerse/config/_task.yml` (currently 50 per task), even
if `EVAL_NUM` is set in the parent shell. Repeat the command with `--seed 1` and
`--seed 2` for the three-seed summary.

Select tasks with `--only task_a,task_b`, `--tasks-file PATH`, or `--limit`.
MiVerse has eight tasks and no capability dimensions. Execution is sequential
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

The environment profile is named `miverse`, but saved results use its
`config_name`, currently `miverse_teleop`:

```text
eval_result/MiVerse/<task>/<policy>/miverse_teleop/<seed>_ckpt_name=...,action_type=.../<run_id>/_result.json
```

```bash
bash scripts/MiVerse/miverse.sh summarize
bash scripts/MiVerse/miverse.sh summarize \
  --ckpt <CHECKPOINT> --action-type joint --env-cfg miverse_teleop
```

The summary script can also run directly:

```bash
python scripts/MiVerse/summarize_result.py \
  --root eval_result/MiVerse --output /tmp/miverse_summary.md
```

`--root` selects the result root containing task directories; otherwise
`MIVERSE_EVAL_ROOT` or `eval_result/MiVerse` is used. `--output` overrides the
default `<root>/_summary.md`. For summary filtering, `--env-cfg` is the result
**config_name directory**, so use `miverse_teleop`, not the profile filename
`miverse`.

Each result configuration and run label, including checkpoint and action type,
forms a separate experiment. Summaries report success rate and score for all
eight tasks and seeds 0, 1, 2, with progress out of 24 task/seed cells. A cell
requires the task's native episode count (currently 50 valid records), and only
the first such records are scored. The latest timestamp containing a result is
used; an incomplete or invalid latest result does not fall back to an older
completed result. Incomplete cells stay blank, and invalid files are reported
in the terminal.

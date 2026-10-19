#!/usr/bin/env python3
"""Aggregate MiVerse eval results into a markdown table.

Layout on disk:
    <task>/<policy>/<embodiment>/<seed>_ckpt_name=...,action_type=.../<timestamp>/_result.json

Rules:
  * MiVerse has 8 tasks. There are no capability dimensions and no
    ``*_random`` splits; each task is scored on its own.
  * Configurations and run labels (including checkpoint and action type) are
    separate experiments. For every (task, experiment, seed) we only read the
    *latest* timestamp folder; an incomplete or invalid latest result never
    falls back to an older completed result.
  * `_result.json` -> `details` holds per-episode entries with `success` (bool)
    and `score` (float in [0, 1]).
  * A cell is filled only when the first 50 episodes are present; otherwise
    the cell stays blank.
  * Each policy table lists seeds 0, 1, 2.
  * The overview lists every policy with at least one completed (task, seed)
    cell. Progress = completed cells / 24 (8 tasks x 3 seeds).
  * Avg SR and Avg Score are the unweighted mean of the 8 tasks. Across seeds
    they are reported as mean ±std. Incomplete policies use the tasks finished
    so far and are marked with dagger in Progress.
  * success_rate = successes / count * 100
    score        = sum(scores) / count * 100
"""

import argparse
import html
import json
import os
import re
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "MIVERSE_EVAL_ROOT",
    os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "eval_result", "MiVerse")),
)
OUTPUT_MD = os.path.join(ROOT, "_summary.md")

SEED_RE = re.compile(r"^(\d+)_")
EXPECTED_SEEDS = [0, 1, 2]
TASKS = []
TASK_EPISODES = {}
TOTAL_REQUIRED = 0


def load_task_spec():
    """Use the same task inventory and native episode counts as evaluation."""
    import yaml

    path = os.path.join(SCRIPT_DIR, "..", "..", "task", "MiVerse", "config", "_task.yml")
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    common_episodes = config["common"]["eval_nums"]
    episodes = {
        task: (settings or {}).get("eval_nums", common_episodes)
        for task, settings in config["tasks"].items()
    }
    if not episodes or any(type(count) is not int or count <= 0 for count in episodes.values()):
        raise SystemExit(f"Invalid MiVerse task episode counts in {path}")
    return episodes


def experiment_label(experiment):
    """Keep every directory-level experiment identifier visible in the output."""
    policy, embodiment, run_info = experiment
    label = f"{policy} / {embodiment} / {run_info or '(unlabeled run)'}"
    return html.escape(label).replace("|", "&#124;").replace("\n", " ")


def list_subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    )


def latest_timestamp_dir(run_dir):
    """Return the latest timestamp subdir that contains a `_result.json`."""
    candidates = []
    for name in list_subdirs(run_dir):
        if os.path.isfile(os.path.join(run_dir, name, "_result.json")):
            candidates.append(name)
    if not candidates:
        return None
    # Timestamp format `YYYY-MM-DD_HH-MM-SS` sorts chronologically as a string.
    return os.path.join(run_dir, sorted(candidates)[-1])


def load_entries(ts_dir):
    """Load (success, score) entries sorted by episode index from a timestamp dir."""
    path = os.path.join(ts_dir, "_result.json")

    def invalid(reason):
        print(f"Warning: ignoring {path}: {reason}", file=sys.stderr)
        return None

    try:
        with open(path) as fh:
            data = json.load(fh)
    except (ValueError, OSError) as exc:
        return invalid(str(exc))
    if not isinstance(data, dict) or not isinstance(data.get("details"), dict):
        return invalid("expected an object containing a details object")
    details = data["details"]
    items = []
    indices = set()
    for key, entry in details.items():
        if not re.fullmatch(r"\d+", key) or not isinstance(entry, dict):
            return invalid(f"invalid episode record {key!r}")
        try:
            index = int(key)
        except ValueError:
            return invalid(f"invalid episode index {key!r}")
        if index in indices:
            return invalid(f"duplicate episode index {index}")
        indices.add(index)
        success = entry.get("success")
        score = entry.get("score")
        if type(success) is not bool:
            return invalid(f"episode {key}: success must be a boolean")
        if type(score) not in (int, float) or not 0 <= score <= 1:
            return invalid(f"episode {key}: score must be a finite number in [0, 1]")
        items.append((index, success, float(score)))
    items.sort(key=lambda x: x[0])
    return [(s, sc) for _, s, sc in items]


def scan_task(task, ckpt=None, action_type=None, env_cfg=None):
    """Return {(experiment, seed): entries}, keeping result configurations apart."""
    task_path = os.path.join(ROOT, task)
    out = {}
    for policy in list_subdirs(task_path):
        policy_path = os.path.join(task_path, policy)
        for embodiment in list_subdirs(policy_path):
            if env_cfg is not None and embodiment != env_cfg:
                continue
            emb_path = os.path.join(policy_path, embodiment)
            for run in list_subdirs(emb_path):
                m = SEED_RE.match(run)
                if not m:
                    continue
                seed = int(m.group(1))
                if seed not in EXPECTED_SEEDS:
                    continue
                run_info = run[m.end():]
                metadata = dict(
                    item.split("=", 1) for item in run_info.split(",") if "=" in item
                )
                if ckpt is not None and metadata.get("ckpt_name") != ckpt:
                    continue
                if action_type is not None and metadata.get("action_type") != action_type:
                    continue
                ts_dir = latest_timestamp_dir(os.path.join(emb_path, run))
                if ts_dir is None:
                    continue
                entries = load_entries(ts_dir)
                if entries is None:
                    continue
                out[((policy, embodiment, run_info), seed)] = entries
    return out


def stats(entries):
    count = len(entries)
    if count == 0:
        return 0, 0.0, 0.0
    successes = sum(1 for s, _ in entries if s)
    score_sum = sum(sc for _, sc in entries)
    return count, successes / count * 100.0, score_sum / count * 100.0


def main(root=None, output=None, ckpt=None, action_type=None, env_cfg=None):
    global ROOT, OUTPUT_MD, TASKS, TASK_EPISODES, TOTAL_REQUIRED
    ROOT = root or ROOT
    OUTPUT_MD = output or os.path.join(ROOT, "_summary.md")
    if not os.path.isdir(ROOT):
        raise SystemExit(
            f"Eval result directory not found: {ROOT}\n"
            "Use --root, set MIVERSE_EVAL_ROOT, or create eval_result/MiVerse."
        )
    TASK_EPISODES = load_task_spec()
    TASKS = list(TASK_EPISODES)
    TOTAL_REQUIRED = len(TASKS) * len(EXPECTED_SEEDS)

    # data[(policy, config_name, run_info)][task][seed] = (success_rate, score)
    data = {}

    def record(policy, task, seed, sr, score):
        data.setdefault(policy, {}).setdefault(task, {})[seed] = (sr, score)

    for task in TASKS:
        for key, entries in scan_task(task, ckpt, action_type, env_cfg).items():
            policy, seed = key
            data.setdefault(policy, {})
            episodes = TASK_EPISODES[task]
            if len(entries) < episodes:
                continue
            _, sr, score = stats(entries[:episodes])
            record(policy, task, seed, sr, score)

    write_markdown(data)
    n_cells = sum(len(seeds) for tasks in data.values() for seeds in tasks.values())
    complete = sum(1 for p in data if policy_is_complete(p, data))
    tested = sum(1 for p in data if policy_progress(p, data)[0] > 0)
    print(
        f"Wrote {len(data)} experiment tables ({n_cells} filled cells, "
        f"{tested} in overview, {complete} complete) to {OUTPUT_MD}"
    )


def mean_with_spread(values, integer=False):
    """Format `mean ±amp` where amp is half the (max - min) range across seeds.

    Returns "" when there is no data so the cell stays blank.
    """
    if not values:
        return ""
    mean = sum(values) / len(values)
    amp = (max(values) - min(values)) / 2.0
    fmt = "{:.0f}" if integer else "{:.2f}"
    return f"{fmt.format(mean)} ±{fmt.format(amp)}"


def task_seed_means(task_data, seeds):
    """Average a task's per-seed (sr, score) over the given seed list.

    Returns None unless every seed has a completed result.
    """
    if not seeds:
        return None
    try:
        values = [task_data[seed] for seed in seeds]
    except KeyError:
        return None
    srs = [sr for sr, _ in values]
    scores = [sc for _, sc in values]
    return sum(srs) / len(srs), sum(scores) / len(scores)


def policy_progress(policy, data):
    """Return (filled_cells, total_cells, percent) for a policy."""
    policy_data = data.get(policy, {})
    filled = sum(
        1 for task in TASKS
        for seed in EXPECTED_SEEDS
        if seed in policy_data.get(task, {})
    )
    pct = filled / TOTAL_REQUIRED * 100.0
    return filled, TOTAL_REQUIRED, pct


def format_progress(filled, total, pct, complete):
    """Format progress for the overview table."""
    if complete:
        return f"{pct:.0f}%"
    return f"{pct:.1f}% † ({filled}/{total})"


def policy_is_complete(policy, data):
    """True when all 24 (8 tasks x 3 seeds) cells are filled."""
    policy_data = data.get(policy, {})
    for task in TASKS:
        if task_seed_means(policy_data.get(task, {}), EXPECTED_SEEDS) is None:
            return False
    return True


def mean_std(values):
    """Return (mean, population std) over values, or None when empty."""
    if not values:
        return None
    mean = sum(values) / len(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_mean_std(stat):
    """Format `mean ±std` to two decimals; blank when no data."""
    if stat is None:
        return ""
    mean, std = stat
    return f"{mean:.2f} ±{std:.2f}"


def seed_task_means(policy_data, seed):
    """Unweighted mean SR/Score of the tasks that have data at this seed."""
    srs = []
    scores = []
    for task in TASKS:
        task_data = policy_data.get(task, {})
        if seed in task_data:
            sr, score = task_data[seed]
            srs.append(sr)
            scores.append(score)
    if not srs:
        return None
    return sum(srs) / len(srs), sum(scores) / len(scores)


def policy_overall(policy_data):
    """Return ((sr_mean, sr_std), (score_mean, score_std)) across seeds.

    Each seed contributes the unweighted mean of the finished tasks. None when
    no seed has data.
    """
    srs = []
    scores = []
    for seed in EXPECTED_SEEDS:
        value = seed_task_means(policy_data, seed)
        if value is None:
            continue
        srs.append(value[0])
        scores.append(value[1])
    if not srs:
        return None
    return mean_std(srs), mean_std(scores)


def build_overview(data):
    """Overview of tested policies, sorted by average SR (descending).

    Avg SR and Avg Score are the unweighted mean of the 8 tasks, shown as
    `mean ±std` across seeds. A cell needs 50 episodes. Incomplete policies
    average the tasks finished so far.
    """
    lines = [
        "## Overview",
        "",
        f"Progress † marks incomplete evaluation ({TOTAL_REQUIRED} = {len(TASKS)} tasks × 3 seeds). "
        f"Avg SR and Avg Score are the unweighted mean of the {len(TASKS)} tasks, "
        "shown as `mean ±std` across seeds. Experiments are sorted by Avg SR.",
        "",
    ]

    header = ["Policy / config_name / run", "Progress", "Avg SR (%)", "Avg Score"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |")

    rows = []
    for policy in data:
        filled, total, pct = policy_progress(policy, data)
        if filled == 0:
            continue
        complete = policy_is_complete(policy, data)
        overall = policy_overall(data[policy])
        sort_key = overall[0][0] if overall else float("-inf")
        cells = [experiment_label(policy), format_progress(filled, total, pct, complete)]
        if overall is None:
            cells.extend(["", ""])
        else:
            cells.append(format_mean_std(overall[0]))
            cells.append(format_mean_std(overall[1]))
        rows.append((sort_key, policy, cells))

    rows.sort(key=lambda r: (-r[0], r[1]))
    for _, _, cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def format_task_row(task, task_data):
    """Build one task row: per-seed cells plus cross-seed average."""
    cells = [f"<nobr>{task}</nobr>"]
    srs = []
    scores = []
    for seed in EXPECTED_SEEDS:
        if seed in task_data:
            sr, score = task_data[seed]
            cells.append(f"{sr:.0f}")
            cells.append(f"{score:.2f}")
            srs.append(sr)
            scores.append(score)
        else:
            cells.append("")
            cells.append("")
    if len(srs) == len(EXPECTED_SEEDS):
        cells.append(mean_with_spread(srs, integer=True))
        cells.append(mean_with_spread(scores))
    else:
        cells.extend(["", ""])
    return cells


def format_avg_row(policy_data):
    """Summary row: unweighted mean of the 8 tasks, per seed and across seeds."""
    cells = [f"**Avg ({len(TASKS)} tasks)**"]
    seed_srs = []
    seed_scores = []
    for seed in EXPECTED_SEEDS:
        means = seed_task_means(policy_data, seed)
        if means is None:
            cells.extend(["", ""])
        else:
            sr, score = means
            cells.append(f"{sr:.1f}")
            cells.append(f"{score:.2f}")
            seed_srs.append(sr)
            seed_scores.append(score)
    if len(seed_srs) == len(EXPECTED_SEEDS):
        cells.append(mean_with_spread(seed_srs, integer=True))
        cells.append(mean_with_spread(seed_scores))
    else:
        cells.extend(["", ""])
    return cells


def write_markdown(data):
    lines = ["# MiVerse Evaluation Summary", ""]
    episode_counts = set(TASK_EPISODES.values())
    count_note = f" ({next(iter(episode_counts))} episodes per task)" if len(episode_counts) == 1 else ""
    lines.extend([
        "Each config_name and run label (including checkpoint and action type) is summarized separately. "
        "Only seeds 0, 1, 2 are counted. Each task needs its native episode count from "
        f"`task/MiVerse/config/_task.yml`{count_note}. Latest incomplete results stay blank; "
        "invalid results are reported in the terminal.",
        "",
    ])
    lines += build_overview(data)

    for policy in sorted(data):
        lines.append(f"## {experiment_label(policy)}")
        lines.append("")

        header = ["Task"]
        for seed in EXPECTED_SEEDS:
            header.append(f"Seed {seed} SR (%)")
            header.append(f"Seed {seed} Score")
        header.append("Avg SR (%)")
        header.append("Avg Score")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |")

        policy_data = data[policy]
        for task in TASKS:
            cells = format_task_row(task, policy_data.get(task, {}))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("| " + " | ".join(format_avg_row(policy_data)) + " |")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_MD)), exist_ok=True)
    with open(OUTPUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def cli(argv=None):
    """Summarize MiVerse results without changing the RoboDojo CLI."""
    parser = argparse.ArgumentParser(
        description="Summarize completed MiVerse evaluation results."
    )
    parser.add_argument("--dataset", choices=("MiVerse",), default="MiVerse", help=argparse.SUPPRESS)
    parser.add_argument(
        "--root",
        help="Result root containing task directories; defaults to MIVERSE_EVAL_ROOT or eval_result/MiVerse.",
    )
    parser.add_argument("--output", help="Markdown output path (default: <root>/_summary.md).")
    parser.add_argument("--ckpt", help="Only include this exact ckpt_name from result directory names.")
    parser.add_argument("--action-type", help="Only include this exact action_type from result directory names.")
    parser.add_argument(
        "--env-cfg", help="Only include this result config_name (e.g. miverse_teleop), not the YAML filename."
    )
    args = parser.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(args.root or os.environ.get(
        "MIVERSE_EVAL_ROOT",
        os.path.join(SCRIPT_DIR, "..", "..", "eval_result", "MiVerse"),
    )))
    output = os.path.abspath(os.path.expanduser(args.output or os.path.join(root, "_summary.md")))
    main(
        root=root, output=output, ckpt=args.ckpt,
        action_type=args.action_type, env_cfg=args.env_cfg,
    )


if __name__ == "__main__":
    cli()

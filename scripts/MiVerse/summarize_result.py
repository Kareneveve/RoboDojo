#!/usr/bin/env python3
"""Summarize MiVerse clean, language (_l), and spatial (_s) evaluations.

Read the latest result for each (task, policy, eval config, seed), using only
25 completed episodes in numeric ``details`` key order. Eval configs under one
policy are collapsed the same way RoboDojo drops embodiment: one table per
policy. When several configs have the same task and seed, configs are walked
in sorted order and a complete result replaces the earlier one. An incomplete
latest run does not replace a completed result from another config. A
task/seed cell needs all 25 entries. Tasks and the three seed averages have
equal weight. Missing seeds leave the three-seed average blank; partial task
coverage is marked with a dagger.

Input: <root>/<task>/<policy>/<config>/<seed>_<run>/<timestamp>/_result.json
Output: <root>/_summary.md. Set MIVERSE_EVAL_ROOT to override the default root.
"""

from datetime import datetime
import json
import math
import os
import re
import statistics
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "MIVERSE_EVAL_ROOT",
    os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "eval_result", "MiVerse")),
)
OUTPUT_MD = os.path.join(ROOT, "_summary.md")
SEED_RE = re.compile(r"^(\d+)_")
EPISODES = 25
EXPECTED_SEEDS = [0, 1, 2]
DIMENSIONS = {"clean": "", "language": "_l", "spatial": "_s"}
TASKS = [
    "place_kettle_on_tv_stand",
    "place_fruit_basket",
    "throw_away_trash",
    "put_object_in_sink",
    "distribute_bowls_to_plates",
    "place_objects_in_dishwasher",
    "place_bread_in_oven",
    "sweep_paper_ball_into_dustbin",
]
TOTAL_REQUIRED = len(TASKS) * len(EXPECTED_SEEDS)
TIMESTAMP_FORMATS = [
    (re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"), "%Y-%m-%d_%H-%M-%S"),
    (re.compile(r"\d{8}_\d{6}"), "%Y%m%d_%H%M%S"),
]


def list_subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    )


def timestamp_key(ts_dir):
    """Compare both supported timestamp formats, including prefixed run names."""
    name = os.path.basename(ts_dir)
    for pattern, fmt in TIMESTAMP_FORMATS:
        match = pattern.search(name)
        if match:
            try:
                return datetime.strptime(match.group(), fmt).timestamp(), ts_dir
            except ValueError:
                pass
    return os.path.getmtime(os.path.join(ts_dir, "_result.json")), ts_dir


def latest_timestamp_dir(run_dir):
    candidates = [
        os.path.join(run_dir, name) for name in list_subdirs(run_dir)
        if os.path.isfile(os.path.join(run_dir, name, "_result.json"))
    ]
    return max(candidates, key=timestamp_key) if candidates else None


def load_entries(ts_dir):
    """Read the first 25 episodes by numeric result key, not by layout_id."""
    path = os.path.join(ts_dir, "_result.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        details = data["details"]
        if not isinstance(details, dict):
            raise ValueError("details must be a mapping")
        indexed = []
        for key, entry in details.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if index >= 0:
                indexed.append((index, entry))
        indexed.sort(key=lambda item: item[0])
        if len({index for index, _ in indexed}) != len(indexed):
            raise ValueError("duplicate episode indices")
        entries = []
        for _, entry in indexed[:EPISODES]:
            success = entry["success"]
            score = float(entry["score"])
            if not isinstance(success, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("success must be bool and score must be in [0, 1]")
            entries.append((success, score))
        return entries
    except (OSError, ValueError, TypeError, KeyError) as exc:
        warnings.warn(f"Skipping invalid result {path}: {exc}", stacklevel=2)
        return None


def scan_task(task, root=None):
    """Return latest entries by (policy, config, seed), without mixing configs."""
    task_path = os.path.join(ROOT if root is None else root, task)
    latest = {}
    for policy in list_subdirs(task_path):
        policy_path = os.path.join(task_path, policy)
        for config in list_subdirs(policy_path):
            config_path = os.path.join(policy_path, config)
            for run in list_subdirs(config_path):
                match = SEED_RE.match(run)
                if not match or int(match.group(1)) not in EXPECTED_SEEDS:
                    continue
                ts_dir = latest_timestamp_dir(os.path.join(config_path, run))
                if ts_dir is None:
                    continue
                key = (policy, config, int(match.group(1)))
                if key not in latest or timestamp_key(ts_dir) > timestamp_key(latest[key]):
                    latest[key] = ts_dir
    # Select before loading: an incomplete/invalid latest run must not silently
    # fall back to an older completed evaluation.
    return {key: load_entries(path) for key, path in latest.items()}


def stats(entries):
    count = len(entries)
    if not count:
        return 0, 0.0, 0.0
    return (count, sum(success for success, _ in entries) / count * 100,
            sum(score for _, score in entries) / count * 100)


def collect_data(root=None):
    """data[policy][dimension][base_task][seed] = (SR, Score).

    Config directories stay separate while picking the latest run, then fold
    into the policy. Sorted config order matches RoboDojo's sorted
    ``(policy, embodiment, seed)`` walk: the last complete config wins.
    """
    data = {}
    for dimension, suffix in DIMENSIONS.items():
        for task in TASKS:
            scanned = scan_task(task + suffix, root)
            for policy, _config, seed in sorted(scanned):
                entries = scanned[(policy, _config, seed)]
                dimension_data = data.setdefault(policy, {}).setdefault(dimension, {})
                if entries is None or len(entries) < EPISODES:
                    continue
                _, sr, score = stats(entries[:EPISODES])
                dimension_data.setdefault(task, {})[seed] = (sr, score)
    return data


def mean_std(values):
    """Population standard deviation, used consistently for all seed averages."""
    if not values:
        return None
    return statistics.mean(values), statistics.pstdev(values)


def format_mean_std(stat):
    return "" if stat is None else f"{stat[0]:.2f} ±{stat[1]:.2f}"


def seed_task_count(dimension_data, seed):
    return sum(seed in dimension_data.get(task, {}) for task in TASKS)


def seed_task_means(dimension_data, seed):
    values = [dimension_data[task][seed] for task in TASKS
              if seed in dimension_data.get(task, {})]
    if not values:
        return None
    return tuple(statistics.mean(items) for items in zip(*values))


def policy_progress(dimension_data):
    filled = sum(seed_task_count(dimension_data, seed) for seed in EXPECTED_SEEDS)
    return filled, TOTAL_REQUIRED, filled / TOTAL_REQUIRED * 100


def policy_is_complete(dimension_data):
    return policy_progress(dimension_data)[0] == TOTAL_REQUIRED


def policy_overall(dimension_data):
    """Mean ±std of the three seed means; require all three seeds to be present."""
    means = [seed_task_means(dimension_data, seed) for seed in EXPECTED_SEEDS]
    if any(value is None for value in means):
        return None
    return tuple(mean_std(values) for values in zip(*means))


def format_progress(dimension_data):
    filled, total, pct = policy_progress(dimension_data)
    marker = "" if filled == total else " †"
    return f"{filled}/{total} ({pct:.1f}%){marker}"


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def append_table(lines, header, rows):
    lines.append("| " + " | ".join(map(markdown_cell, header)) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for cells in rows:
        lines.append("| " + " | ".join(map(markdown_cell, cells)) + " |")
    lines.append("")


def ranking_rows(data, dimension):
    """Rank every policy on one dimension, highest success rate first."""
    ranked = []
    unranked = []
    for policy, dimensions in data.items():
        dimension_data = dimensions.get(dimension, {})
        overall = policy_overall(dimension_data)
        if overall is None:
            unranked.append((policy, dimension_data))
        else:
            ranked.append((policy, dimension_data, overall))
    ranked.sort(key=lambda item: (-item[2][0][0], -item[2][1][0], item[0]))
    unranked.sort(key=lambda item: item[0])
    rows = []
    for index, (policy, dimension_data, overall) in enumerate(ranked, start=1):
        marker = "" if policy_is_complete(dimension_data) else " †"
        sr, score = overall
        rows.append([
            index,
            policy,
            format_mean_std(sr) + marker,
            format_mean_std(score) + marker,
            format_progress(dimension_data),
        ])
    for policy, dimension_data in unranked:
        rows.append(["", policy, "", "", format_progress(dimension_data)])
    return rows


def section_order(data):
    """Detail sections follow the clean ranking, then policies with no clean score."""
    def sort_key(item):
        policy, dimensions = item
        overall = policy_overall(dimensions.get("clean", {}))
        if overall is None:
            return (1, 0.0, 0.0, policy)
        sr, score = overall
        return (0, -sr[0], -score[0], policy)

    return sorted(data.items(), key=sort_key)


def build_overview(data):
    lines = [
        "## Policy ranking by dimension", "",
        "Each dimension ranks every policy from high to low by success rate. "
        "Score breaks ties. The value is the equally weighted mean ± population "
        "std of the seed 0/1/2 task averages. A blank rank means that dimension "
        "has no three-seed average. Policy names are shown without the eval-config suffix.", "",
    ]
    for dimension in DIMENSIONS:
        lines.extend([f"### {dimension}", ""])
        append_table(
            lines,
            ["Rank", "Policy", "SR (%)", "Score (%)", "Progress"],
            ranking_rows(data, dimension),
        )
    return lines


def format_task_row(task, task_data):
    cells = [task]
    values = []
    for seed in EXPECTED_SEEDS:
        if seed in task_data:
            sr, score = task_data[seed]
            cells.extend([f"{sr:.2f}", f"{score:.2f}"])
            values.append((sr, score))
        else:
            cells.extend(["", ""])
    if len(values) == len(EXPECTED_SEEDS):
        cells.extend(format_mean_std(mean_std(items)) for items in zip(*values))
    else:
        cells.extend(["", ""])
    return cells


def format_avg_row(dimension_data):
    cells = ["**Avg across tasks**"]
    for seed in EXPECTED_SEEDS:
        means = seed_task_means(dimension_data, seed)
        if means is None:
            cells.extend(["", ""])
        else:
            marker = "" if seed_task_count(dimension_data, seed) == len(TASKS) else " †"
            cells.extend(f"{value:.2f}{marker}" for value in means)
    overall = policy_overall(dimension_data)
    if overall is None:
        cells.extend(["", ""])
    else:
        marker = "" if policy_is_complete(dimension_data) else " †"
        cells.extend(format_mean_std(value) + marker for value in overall)
    return cells


def write_markdown(data, output_path=None):
    lines = [
        "# MiVerse Evaluation Summary", "",
        "- Dimensions: **clean** = original task; **language** = `_l`; **spatial** = `_s`.",
        f"- Each task/seed uses only the first **{EPISODES}** completed episodes, "
        "sorted by the numeric `details` key, not `layout_id`. Later episodes are ignored.",
        f"- Use the latest result per task/policy/eval config/seed, then keep one "
        f"result per policy. Configs are taken in sorted order; a complete result "
        f"replaces an earlier config, and fewer than {EPISODES} episodes does not. "
        "No fallback to an older run inside the same config.",
        "- SR = successes / 25 × 100; Score = sum of episode scores / 25 × 100.",
        "- Each seed averages its completed tasks equally. The three-seed average "
        "is blank unless seed 0, 1, and 2 all have completed task results.",
        "- **†** means incomplete task coverage: averages use only completed tasks "
        "and are provisional. Only 24/24 cells gives a complete dimension score.",
        "- All `±` values are population standard deviations across the three seeds.",
        "- Each dimension ranks every policy from high to low by success rate. "
        "Headings use the policy name only, without the eval-config suffix.", "",
    ]
    lines += build_overview(data)
    for policy, dimensions in section_order(data):
        lines.extend([f"## {markdown_cell(policy)}", ""])
        for dimension, suffix in DIMENSIONS.items():
            dimension_data = dimensions.get(dimension, {})
            coverage = ", ".join(
                f"seed {seed}: {seed_task_count(dimension_data, seed)}/{len(TASKS)} tasks"
                for seed in EXPECTED_SEEDS
            )
            lines.extend([f"### {dimension}", "", f"Completed — {coverage}.", ""])
            header = ["Task"]
            for seed in EXPECTED_SEEDS:
                header.extend([f"Seed {seed} SR (%)", f"Seed {seed} Score (%)"])
            header.extend(["3-seed SR (%) mean ±std", "3-seed Score (%) mean ±std"])
            rows = [format_task_row(task + suffix, dimension_data.get(task, {})) for task in TASKS]
            rows.append(format_avg_row(dimension_data))
            append_table(lines, header, rows)
    with open(OUTPUT_MD if output_path is None else output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(
            f"Eval result directory not found: {ROOT}\n"
            "Set MIVERSE_EVAL_ROOT or run with eval_result/MiVerse present."
        )
    data = collect_data()
    write_markdown(data)
    filled = sum(policy_progress(tasks)[0] for dims in data.values() for tasks in dims.values())
    complete = sum(policy_is_complete(tasks) for dims in data.values() for tasks in dims.values())
    print(f"Wrote {len(data)} policies, {filled} completed task/seed cells, "
          f"{complete} complete dimensions to {OUTPUT_MD}")


if __name__ == "__main__":
    main()

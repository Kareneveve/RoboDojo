#!/usr/bin/env python3
"""Partition benchmark tasks using measured runtimes when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_profile import BENCHMARKS

# Preserve RoboDojo's measured arx_x5 weights and partition algorithm.
ROBODOJO_RUNTIME_WEIGHTS = {
    "imitate_sorting_sequence/arx_x5": 10596,
    "press_by_number/arx_x5": 2955,
    "sweep_blocks_random/arx_x5": 2392,
    "align_blocks/arx_x5": 2013,
    "plug_in_charger/arx_x5": 1777,
    "fold_clothes/arx_x5": 1435,
    "pour_by_language/arx_x5": 9448,
    "fill_egg_holder/arx_x5": 3063,
    "pack_objects_into_box/arx_x5": 2725,
    "pour_balls_into_vase/arx_x5": 2509,
    "solve_equation/arx_x5": 1867,
    "insert_key/arx_x5": 1396,
    "play_tic_tac_toe/arx_x5": 9323,
    "arrange_largest_number_random/arx_x5": 3101,
    "match_and_pick_from_conveyor/arx_x5": 2915,
    "sort_nesting_dolls_by_size/arx_x5": 2377,
    "stack_blocks_by_language/arx_x5": 1733,
    "store_laptop_and_headphones/arx_x5": 1715,
    "fasten_screws/arx_x5": 7876,
    "make_toast_random/arx_x5": 3775,
    "swap_blocks/arx_x5": 2941,
    "stack_bowls/arx_x5": 1806,
    "pour_liquid_into_cup_random/arx_x5": 1798,
    "hang_mugs/arx_x5": 1734,
    "general_pickup/arx_x5": 1201,
    "play_stacking_toy/arx_x5": 5416,
    "make_kong/arx_x5": 3818,
    "pick_from_conveyor_by_image/arx_x5": 3150,
    "pack_objects_into_box_random/arx_x5": 2917,
    "play_Xylophone/arx_x5": 2174,
    "hang_mugs_random/arx_x5": 2047,
    "push_T_random/arx_x5": 1593,
    "classify_objects_by_language/arx_x5": 4955,
    "cover_blocks/arx_x5": 3837,
    "make_toast/arx_x5": 3781,
    "pour_liquid_into_cup/arx_x5": 2543,
    "arrange_largest_number/arx_x5": 2346,
    "store_laptop_and_headphones_random/arx_x5": 1981,
    "swap_T/arx_x5": 1746,
    "classify_objects/arx_x5": 4881,
    "organize_table/arx_x5": 4104,
    "store_tools_in_toolbox/arx_x5": 3783,
    "sweep_blocks/arx_x5": 2170,
    "fold_clothes_random/arx_x5": 1858,
    "stack_blocks_random/arx_x5": 1768,
    "deposit_coin/arx_x5": 1373,
    "stack_blocks/arx_x5": 1252,
    "build_tower/arx_x5": 4750,
    "fill_pen_holder/arx_x5": 4698,
    "sort_nesting_dolls_by_size_random/arx_x5": 3021,
    "put_bottles_into_dustbin/arx_x5": 2887,
    "insert_tubes/arx_x5": 2296,
    "stack_bowls_random/arx_x5": 2146,
    "push_T/arx_x5": 1311,
}


def score(groups: list[list[dict]]) -> tuple[int, int]:
    loads = [sum(item["seconds"] for item in group) for group in groups]
    return max(loads), max(loads) - min(loads)


def partition(tasks: list[dict], group_count: int) -> list[list[dict]]:
    groups = [[] for _ in range(group_count)]
    loads = [0] * group_count
    for task in tasks:
        idx = min(range(group_count), key=lambda i: (loads[i], i))
        groups[idx].append(task)
        loads[idx] += task["seconds"]

    while True:
        current = score(groups)
        loads = [sum(item["seconds"] for item in group) for group in groups]
        src_idx = max(range(len(groups)), key=lambda i: (loads[i], i))
        best = None

        for task_idx, task in enumerate(groups[src_idx]):
            for dst_idx in range(len(groups)):
                if dst_idx == src_idx:
                    continue
                candidate = [list(group) for group in groups]
                candidate[src_idx].pop(task_idx)
                candidate[dst_idx].append(task)
                candidate_score = score(candidate)
                if candidate_score < current and (best is None or candidate_score < best[0]):
                    best = (candidate_score, candidate)

        for task_idx, task in enumerate(groups[src_idx]):
            for dst_idx in range(len(groups)):
                if dst_idx == src_idx:
                    continue
                for other_idx, other in enumerate(groups[dst_idx]):
                    candidate = [list(group) for group in groups]
                    candidate[src_idx][task_idx] = other
                    candidate[dst_idx][other_idx] = task
                    candidate_score = score(candidate)
                    if candidate_score < current and (best is None or candidate_score < best[0]):
                        best = (candidate_score, candidate)

        if best is None:
            break
        groups = best[1]

    for group in groups:
        group.sort(key=lambda item: (-item["seconds"], item["task"]))
    return groups



def build_assignment(bench, env_cfg, task_names, policy_gpus, env_gpus):
    if bench not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {bench}")
    if not policy_gpus or len(policy_gpus) != len(env_gpus):
        raise ValueError("Non-empty policy/env GPU lists must have equal lengths")
    weights = ROBODOJO_RUNTIME_WEIGHTS if (bench, env_cfg) == ("RoboDojo", "arx_x5") else None
    if weights is not None:
        missing = [f"{task}/{env_cfg}" for task in task_names if f"{task}/{env_cfg}" not in weights]
        if missing:
            raise ValueError("Missing runtime entries: " + ", ".join(missing))
        tasks = [
            {"task": task, "key": f"{task}/{env_cfg}", "seconds": weights[f"{task}/{env_cfg}"]}
            for task in task_names
        ]
        tasks.sort(key=lambda item: (-item["seconds"], item["task"]))
        groups = partition(tasks, len(policy_gpus))
    else:
        # Without measured runtimes, distribute whole tasks by count.
        groups = [[] for _ in policy_gpus]
        for index, task in enumerate(task_names):
            groups[index % len(groups)].append({"task": task})
    assignments = []
    for index, (policy_gpu, env_gpu, group) in enumerate(zip(policy_gpus, env_gpus, groups)):
        assignment = {
            "worker": index,
            "policy_gpu": policy_gpu,
            "env_gpu": env_gpu,
            "tasks": group,
            "load": f"{len(group)} tasks",
        }
        if weights is not None:
            assignment["total_seconds"] = sum(item["seconds"] for item in group)
            assignment["load"] = f"{assignment['total_seconds']}s"
        assignments.append(assignment)
    return {
        "benchmark": bench,
        "env_cfg": env_cfg,
        "weights_source": "embedded" if weights is not None else "task_count",
        "groups": assignments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=BENCHMARKS, required=True)
    parser.add_argument("--env-cfg", required=True)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--policy-gpu-ids", required=True)
    parser.add_argument("--env-gpu-ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        tasks = [line.strip() for line in args.tasks_file.read_text().splitlines() if line.strip()]
        policy_gpus = args.policy_gpu_ids.split(",")
        env_gpus = args.env_gpu_ids.split(",")
        if any(not gpu.strip() for gpu in policy_gpus + env_gpus):
            raise ValueError("GPU ids must not be empty")
        assignment = build_assignment(args.bench, args.env_cfg, tasks, policy_gpus, env_gpus)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(assignment, indent=2) + "\n")
    print(json.dumps(assignment, indent=2))


if __name__ == "__main__":
    main()

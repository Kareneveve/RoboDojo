#!/usr/bin/env python3
"""List MiVerse tasks without importing Isaac-dependent task modules."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]


def build_inventory() -> dict[str, Any]:
    task_dir = ROOT_DIR / "task" / "MiVerse" / "tasks"
    config_dir = ROOT_DIR / "task" / "MiVerse" / "config"
    tasks = []
    for path in sorted(task_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        config_path = config_dir / f"{name}.yml"
        record = {
            "name": name,
            "module": str(path.relative_to(ROOT_DIR)),
            "class_name": name,
            "class_exists": name in classes,
            "config": str(config_path.relative_to(ROOT_DIR)) if config_path.is_file() else None,
            "config_exists": config_path.is_file(),
        }
        record["runnable"] = bool(record["class_exists"] and record["config_exists"])
        tasks.append(record)
    task_names = {task["name"] for task in tasks}
    config_names = {path.stem for path in config_dir.glob("*.yml") if not path.name.startswith("_")}
    config_only = sorted(config_names - task_names)
    return {
        "benchmark": "MiVerse",
        "root": str(ROOT_DIR),
        "task_dir": str(task_dir.relative_to(ROOT_DIR)),
        "config_dir": str(config_dir.relative_to(ROOT_DIR)),
        "counts": {
            "tasks": len(tasks),
            "runnable": sum(task["runnable"] for task in tasks),
            "missing_config": sum(not task["config_exists"] for task in tasks),
            "missing_class": sum(not task["class_exists"] for task in tasks),
            "config_only": len(config_only),
        },
        "tasks": tasks,
        "config_only": config_only,
    }


def _print_markdown(inventory: dict[str, Any]) -> None:
    print("| Task | Runnable | Config | Issue |")
    print("| --- | --- | --- | --- |")
    for task in inventory["tasks"]:
        issues = []
        if not task["class_exists"]:
            issues.append("missing exported class")
        if not task["config_exists"]:
            issues.append("missing config")
        print(
            f"| `{task['name']}` | {'yes' if task['runnable'] else 'no'} | "
            f"`{task['config'] or '-'}` | {', '.join(issues) or '-'} |"
        )
    if inventory["config_only"]:
        print("\nConfig-only entries:")
        for name in inventory["config_only"]:
            print(f"- `{name}`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MiVerse",), default="MiVerse")
    parser.add_argument("--format", choices=("plain", "json", "markdown"), default="plain")
    parser.add_argument("--only-runnable", action="store_true", help="Only print runnable names in plain output.")
    parser.add_argument("--dimension", action="append", default=[], help="Only all is supported for MiVerse.")
    parser.add_argument("--resolve-dimensions", action="store_true", help="Validate and print all.")
    parser.add_argument("--check", action="store_true", help="Check task modules and configurations.")
    args = parser.parse_args()

    if any(item.strip().lower() != "all" for value in args.dimension for item in value.split(",")):
        parser.error("MiVerse has no capability dimension groups; use --dimension all")
    if args.resolve_dimensions:
        print("all")
        return 0

    inventory = build_inventory()
    if args.format == "json":
        print(json.dumps(inventory, indent=2, sort_keys=True))
    elif args.format == "markdown":
        _print_markdown(inventory)
    else:
        for task in inventory["tasks"]:
            if not args.only_runnable or task["runnable"]:
                print(task["name"])

    if args.check:
        issues = [f"Task {task['name']} is not runnable" for task in inventory["tasks"] if not task["runnable"]]
        issues.extend(f"Config {name} has no task module" for name in inventory["config_only"])
        if not inventory["tasks"]:
            issues.append("No MiVerse task modules found")
        for issue in issues:
            print(f"[ERROR] {issue}", file=sys.stderr)
        if issues:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

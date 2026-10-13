"""Dynamic task loading with an explicit benchmark namespace."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from env.benchmark import BenchmarkPaths
from env.global_configs import BENCHMARK


def load_task_class(task_name: str, benchmark: str = BENCHMARK):
    benchmark = BenchmarkPaths(benchmark).name
    module_name = f"task.{benchmark}.tasks.{task_name}"
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"Could not resolve task {task_name!r} in benchmark {benchmark!r}")
    task_module = importlib.import_module(module_name)
    if not hasattr(task_module, task_name):
        raise ModuleNotFoundError(f"Could not resolve task class {task_name!r}")
    return task_name, getattr(task_module, task_name)


def task_config_path(config_dir: str | Path, task_name: str) -> Path:
    return Path(config_dir) / f"{task_name}.yml"

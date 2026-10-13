"""Backward-compatible imports for the original RoboDojo task loader."""

from task.task_registry import load_task_class, task_config_path

__all__ = ["load_task_class", "task_config_path"]

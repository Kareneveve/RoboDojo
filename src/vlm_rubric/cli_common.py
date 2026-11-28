"""Shared CLI argument builders, validation, and summary helpers for VLM Rubric scripts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .client import VLMJudgeClient
from .judge import extract_criterion_ids
from .rubrics import configured_rubric_tasks, get_rubric_criteria_prompt


def add_judge_api_args(parser: argparse.ArgumentParser, *, default_retries: int = 3) -> None:
    """Attach OpenAI-compatible VLM Judge API options to an ArgumentParser."""
    group = parser.add_argument_group("VLM Judge API options")
    group.add_argument(
        "--model",
        default=None,
        help="VLM Judge model name (default: VLM_JUDGE_MODEL / QWEN_VLM_MODEL or qwen3.8-max)",
    )
    group.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: VLM_JUDGE_BASE_URL / DASHSCOPE_BASE_URL / DashScope)",
    )
    group.add_argument(
        "--api-key",
        default=None,
        help="API key (otherwise VLM_JUDGE_API_KEY, QWEN_VLM_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY)",
    )
    group.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_tokens",
        type=int,
        default=1800,
        help="Maximum generated tokens per episode (default: 1800)",
    )
    group.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request HTTP timeout in seconds (default: 300)",
    )
    group.add_argument(
        "--retries",
        type=int,
        default=default_retries,
        help=f"Retry attempts for transient API failures (default: {default_retries})",
    )


def add_eval_args(
    parser: argparse.ArgumentParser,
    *,
    default_workers: int = 4,
    include_max_episodes: bool = True,
    resume_default: bool = False,
) -> None:
    """Attach common episode sampling, concurrency, and `_result.json` update options."""
    group = parser.add_argument_group("Rubric evaluation & result options")
    group.add_argument(
        "--sample-fraction",
        type=float,
        default=0.10,
        help="Fraction of source frames sampled per episode (default: 0.10, every 10th frame)",
    )
    if include_max_episodes:
        group.add_argument("--max-episodes", type=int, default=None, help="Optional episode limit for a smoke run")
    group.add_argument(
        "--view-policy",
        choices=("strict", "available"),
        default="strict",
        help="Require all three views (strict), or judge available views with coverage metadata",
    )
    group.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Concurrent episode judge workers per run (default: {default_workers})",
    )
    group.add_argument(
        "--only-sim-success",
        action="store_true",
        help="Only submit episodes with sim_success=True to the VLM Judge and skip sim_fail episodes",
    )
    if resume_default:
        group.add_argument(
            "--no-resume",
            action="store_true",
            help="Ignore existing rubric_eval.json checkpoints and re-judge episodes from scratch",
        )
    else:
        group.add_argument(
            "--resume",
            action="store_true",
            help="Reuse completed episodes from an existing rubric_eval.json and checkpoint after each episode",
        )
    group.add_argument(
        "--no-update-result",
        action="store_true",
        help="Write rubric_eval.json/md only without modifying _result.json",
    )
    group.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating _result.sim_backup.json before modifying _result.json in-place",
    )
    group.add_argument(
        "--skip-update-on-error",
        action="store_true",
        help="Do not overwrite _result.json if any episode hits a judge/API error",
    )


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Attach structured logging options."""
    group = parser.add_argument_group("Logging options")
    group.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory to write timestamped log files (also updates latest.log)",
    )
    group.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Explicit log file path (overrides --log-dir)",
    )
    group.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console/file log verbosity (default: INFO)",
    )
    group.add_argument(
        "--quiet",
        action="store_true",
        help="Only show warnings/errors on console while keeping full INFO logs in --log-file/--log-dir",
    )


def validate_common_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate shared numeric CLI arguments."""
    if not 0 < args.sample_fraction <= 1:
        parser.error("--sample-fraction must be greater than 0 and at most 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if getattr(args, "max_episodes", None) is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")


def build_judge_client(
    args: argparse.Namespace,
    client_cls: type[VLMJudgeClient] = VLMJudgeClient,
) -> VLMJudgeClient:
    """Construct a VLMJudgeClient instance from parsed CLI arguments."""
    return client_cls(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        retries=args.retries,
    )


def summarize_episodes(episodes: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Count episode judge statuses and verdicts consistently across all CLIs."""
    counts = {"total": 0, "completed": 0, "pass": 0, "fail": 0, "uncertain": 0, "skipped": 0, "error": 0}
    for entry in episodes.values():
        counts["total"] += 1
        status = entry.get("judge_status", entry.get("review_status"))
        if status == "skipped":
            counts["skipped"] += 1
        elif status != "completed":
            counts["error"] += 1
        else:
            counts["completed"] += 1
            verdict = str(entry.get("verdict", entry.get("vlm_verdict", "uncertain"))).lower()
            counts[verdict if verdict in {"pass", "fail", "uncertain"} else "uncertain"] += 1
    return counts


def format_rubric_catalog() -> str:
    """Return a clean human-readable catalog of configured task rubrics and criterion IDs."""
    tasks = configured_rubric_tasks()
    lines = [f"Configured VLM Rubric tasks ({len(tasks)}):"]
    for task in tasks:
        criteria = extract_criterion_ids(get_rubric_criteria_prompt(task))
        lines.append(f"  - {task:<28s} criteria=[{', '.join(criteria)}]")
    return "\n".join(lines)


__all__ = [
    "add_eval_args",
    "add_judge_api_args",
    "add_logging_args",
    "build_judge_client",
    "format_rubric_catalog",
    "summarize_episodes",
    "validate_common_args",
]

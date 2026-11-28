#!/usr/bin/env python3
"""Run VLM Rubric evaluation over RoboDojo AGP showcase runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_rubric import (
    VLMJudgeClient,
    configure_rubric_logging,
    evaluate_run,
    format_duration,
    get_rubric_criteria_prompt,
)
from src.vlm_rubric.cli_common import (
    add_eval_args,
    add_judge_api_args,
    add_logging_args,
    build_judge_client,
    summarize_episodes,
    validate_common_args,
)


def _run_dir(task_dir: Path) -> Path:
    candidates = sorted(task_dir.glob("AGP/arx_x5/*/*"))
    if not candidates:
        raise FileNotFoundError(f"No AGP run found under {task_dir}")
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade RoboDojo AGP showcase runs via VLM Rubric API")
    parser.add_argument(
        "--agp-root",
        type=Path,
        default=Path(os.environ.get("ROBODOJO_AGP_ROOT", REPO_ROOT.parent / "robodojo_video" / "RoboDojo_AGP_top10")),
        help=(
            "Root directory of AGP showcase videos "
            "(default: $ROBODOJO_AGP_ROOT or ../robodojo_video/RoboDojo_AGP_top10)"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "eval_vlm_agp")
    parser.add_argument("--task", action="append", help="Task to evaluate. Repeat for multiple tasks. Default is all.")
    add_judge_api_args(parser, default_retries=3)
    add_eval_args(parser, default_workers=4, include_max_episodes=True, resume_default=False)
    add_logging_args(parser)

    args = parser.parse_args(argv)
    validate_common_args(parser, args)

    logger, log_path = configure_rubric_logging(
        log_file=args.log_file,
        log_dir=args.log_dir,
        run_name="rubric_agp",
        level=args.log_level,
        quiet=args.quiet,
    )
    if log_path is not None:
        logger.info("log file: %s", log_path)

    if not args.agp_root.exists():
        parser.error(f"AGP root does not exist: {args.agp_root}")

    tasks = sorted(path for path in args.agp_root.iterdir() if path.is_dir())
    if args.task:
        selected = set(args.task)
        tasks = [path for path in tasks if path.name in selected]
    if not tasks:
        parser.error("No matching AGP task directories found")

    configured_tasks = [path for path in tasks if get_rubric_criteria_prompt(path.name)]
    skipped_tasks = [path for path in tasks if not get_rubric_criteria_prompt(path.name)]
    for task_dir in skipped_tasks:
        logger.info("[rubric-agp] task=%s skipped=no task-specific rubric configured", task_dir.name)
    if not configured_tasks:
        return 0

    client = build_judge_client(args, client_cls=VLMJudgeClient)

    failures = 0
    for task_dir in configured_tasks:
        run_dir = _run_dir(task_dir)
        report = evaluate_run(
            run_dir,
            output_dir=args.output_root / task_dir.name,
            task=task_dir.name,
            sample_fraction=args.sample_fraction,
            max_episodes=args.max_episodes,
            view_policy=args.view_policy,
            update_result=not args.no_update_result,
            create_backup=not args.no_backup,
            skip_update_on_error=args.skip_update_on_error,
            workers=args.workers,
            client=client,
            logger=logger,
            resume=args.resume,
            only_sim_success=args.only_sim_success,
        )
        counts = summarize_episodes(report["episodes"])
        if args.view_policy == "strict":
            failures += sum(
                entry.get("judge_status") == "error" and entry.get("error") != "missing required camera video"
                for entry in report["episodes"].values()
            )
        else:
            failures += counts["error"]
        if report.get("result_update", {}).get("status") == "error":
            failures += 1
            logger.error(
                "[rubric-agp] task=%s result update failed: %s",
                task_dir.name,
                report["result_update"].get("error", "unknown error"),
            )
        eligible = counts["total"] - counts["skipped"]
        sampling = report.get("sampling", {})
        logger.info(
            "[rubric-agp] task=%s completed=%d/%d P=%d F=%d U=%d skipped=%d "
            "sample_fraction=%s frame_stride=%s elapsed=%s",
            task_dir.name,
            counts["completed"],
            eligible,
            counts["pass"],
            counts["fail"],
            counts["uncertain"],
            counts["skipped"],
            sampling.get("sample_fraction"),
            sampling.get("frame_stride"),
            format_duration(report.get("elapsed_s", 0.0)),
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

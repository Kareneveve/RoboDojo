#!/usr/bin/env python3
"""Run VLM-as-a-Judge rubric evaluation over a single RoboDojo run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_rubric import (
    RUBRIC_REPORT_JSON,
    RUBRIC_REPORT_MD,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade one RoboDojo run directory against its VLM Rubric using synchronized camera views."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory containing episode videos and optionally _result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Optional writable directory for {RUBRIC_REPORT_JSON} and {RUBRIC_REPORT_MD} (default: inside --run-dir)",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        default=None,
        help="Optional alternate output path for the graded result JSON instead of modifying _result.json in-place",
    )
    parser.add_argument("--task", help="Task name. Defaults to the task directory inferred from --run-dir")
    parser.add_argument(
        "--rubric-file",
        type=Path,
        help="Optional UTF-8 task rubric appended to the built-in rubric criteria",
    )
    add_judge_api_args(parser, default_retries=3)
    add_eval_args(parser, default_workers=4, include_max_episodes=True, resume_default=False)
    add_logging_args(parser)

    args = parser.parse_args(argv)
    validate_common_args(parser, args)

    rubric = args.rubric_file.read_text(encoding="utf-8") if args.rubric_file else None
    resolved_task = args.task
    if not resolved_task:
        run_path = args.run_dir.expanduser().resolve()
        resolved_task = run_path.parents[3].name if len(run_path.parents) > 3 else "unknown"

    logger, log_path = configure_rubric_logging(
        log_file=args.log_file,
        log_dir=args.log_dir,
        run_name=f"judge_{resolved_task}",
        level=args.log_level,
        quiet=args.quiet,
    )
    if log_path is not None:
        logger.info("log file: %s", log_path)

    rubric_configured = rubric is not None or bool(get_rubric_criteria_prompt(resolved_task))
    client = build_judge_client(args, client_cls=VLMJudgeClient) if rubric_configured else None

    report = evaluate_run(
        args.run_dir,
        output_dir=args.output_dir,
        task=resolved_task,
        sample_fraction=args.sample_fraction,
        max_episodes=args.max_episodes,
        rubric=rubric,
        view_policy=args.view_policy,
        update_result=not args.no_update_result,
        create_backup=not args.no_backup,
        result_output_path=args.result_output,
        skip_update_on_error=args.skip_update_on_error,
        workers=args.workers,
        client=client,
        logger=logger,
        resume=args.resume,
        only_sim_success=args.only_sim_success,
    )
    counts = summarize_episodes(report["episodes"])
    eligible = counts["total"] - counts["skipped"]
    logger.info("wrote %s/%s and %s", report["report_dir"], RUBRIC_REPORT_JSON, RUBRIC_REPORT_MD)
    logger.info(
        "summary task=%s completed=%d/%d P=%d F=%d U=%d skipped=%d errors=%d "
        "sample_fraction=%s frame_stride=%s judge_model=%s elapsed=%s",
        report["task"],
        counts["completed"],
        eligible,
        counts["pass"],
        counts["fail"],
        counts["uncertain"],
        counts["skipped"],
        counts["error"],
        report["sampling"]["sample_fraction"],
        report["sampling"]["frame_stride"],
        report["judge_model"],
        format_duration(report.get("elapsed_s", 0.0)),
    )
    result_update = report.get("result_update", {})
    result_update_failed = result_update.get("status") == "error"
    if result_update_failed:
        logger.error("result update failed: %s", result_update.get("error", "unknown error"))
    return 1 if counts["error"] or result_update_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

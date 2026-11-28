#!/usr/bin/env python3
"""Batch-evaluate RoboDojo eval_result runs with VLM Rubrics and manage `_result.json` updates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_rubric import (
    RUBRIC_REPORT_JSON,
    RUBRIC_REPORT_MD,
    SIM_BACKUP_FILENAME,
    TASK_RUBRICS,
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
    format_rubric_catalog,
    summarize_episodes,
    validate_common_args,
)
from src.vlm_rubric.result_store import atomic_write_json


@dataclass(frozen=True)
class DiscoveredRun:
    task: str
    run_dir: Path
    rel_path: str
    total_episodes: int
    sim_success_episodes: int
    video_count: int
    already_graded: bool
    has_backup: bool
    sim_success_rate: float | None
    current_success_rate: float | None
    sim_score: float | None
    current_score: float | None


SEED_RE = re.compile(r"^(\d+)_")


def _find_task_index(rel_parts: tuple[str, ...]) -> int | None:
    for idx, part in enumerate(rel_parts):
        if part in TASK_RUBRICS:
            return idx
    return None


def _latest_group_key(rel_parts: tuple[str, ...], task_idx: int) -> tuple[Any, ...]:
    """Mirror `scripts/internal/summarize_result.py` grouping: (prefix, task, policy, embodiment, seed)."""
    tail = rel_parts[task_idx:]
    if len(tail) >= 5:
        task, policy, embodiment, seed_dir = tail[0], tail[1], tail[2], tail[3]
        seed_match = SEED_RE.match(seed_dir)
        seed_key: int | str = int(seed_match.group(1)) if seed_match else seed_dir
        return (rel_parts[:task_idx], task, policy, embodiment, seed_key)
    return (rel_parts[:-1],)


def _is_cleanly_graded(payload: dict[str, Any]) -> bool:
    if payload.get("result_source") not in {"simulator_plus_vlm_rubric", "simulator_plus_vlm_posthoc"}:
        return False
    meta = payload.get("rubric_eval") or payload.get("vlm_adjudication")
    if not isinstance(meta, dict):
        return False
    counts = meta.get("counts")
    if not isinstance(counts, dict):
        return False
    err_count = int(counts.get("judge_error", counts.get("review_error", 0)) or 0)
    missing_count = int(counts.get("missing", 0) or 0)
    return err_count == 0 and missing_count == 0


def discover_eval_runs(
    eval_root: Path,
    *,
    tasks: set[str] | None = None,
    match: str | None = None,
    latest_only: bool = True,
) -> tuple[list[DiscoveredRun], int, int]:
    """Find Rubric-configured run directories containing `_result.json` under `eval_root`.

    When `latest_only=True` (default), follows `scripts/internal/summarize_result.py`:
    for each `(task, policy, embodiment, seed)`, only the latest timestamp folder
    (lexicographically largest directory name) with a valid `_result.json` is selected.

    Returns `(runs, skipped_non_rubric, skipped_older)`.
    """
    resolved_root = eval_root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {eval_root}")

    candidates: list[tuple[tuple[Any, ...], str, DiscoveredRun]] = []
    skipped_non_rubric = 0
    for result_file in sorted(resolved_root.rglob("_result.json")):
        run_dir = result_file.parent
        rel = run_dir.relative_to(resolved_root)
        task_idx = _find_task_index(rel.parts)
        if task_idx is None:
            skipped_non_rubric += 1
            continue
        task = rel.parts[task_idx]
        if not get_rubric_criteria_prompt(task):
            skipped_non_rubric += 1
            continue
        if tasks and task not in tasks:
            continue
        rel_str = rel.as_posix()
        if match and match not in rel_str:
            continue

        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        details = payload.get("details")
        if not isinstance(details, dict):
            continue

        total_eps = len(details)
        sim_succ = sum(
            1
            for entry in details.values()
            if isinstance(entry, dict) and bool(entry.get("sim_success", entry.get("success", False)))
        )
        video_count = sum(1 for _ in run_dir.glob("episode_*_cam_*.mp4"))
        run_item = DiscoveredRun(
            task=task,
            run_dir=run_dir,
            rel_path=rel_str,
            total_episodes=total_eps,
            sim_success_episodes=sim_succ,
            video_count=video_count,
            already_graded=_is_cleanly_graded(payload),
            has_backup=(run_dir / SIM_BACKUP_FILENAME).exists(),
            sim_success_rate=payload.get("sim_success_rate", payload.get("success_rate")),
            current_success_rate=payload.get("success_rate"),
            sim_score=payload.get("sim_score", payload.get("score")),
            current_score=payload.get("score"),
        )
        candidates.append((_latest_group_key(rel.parts, task_idx), run_dir.name, run_item))

    if not latest_only:
        return [item for _, _, item in candidates], skipped_non_rubric, 0

    selected: dict[tuple[Any, ...], tuple[str, DiscoveredRun]] = {}
    skipped_older = 0
    for group_key, ts_name, run_item in candidates:
        previous = selected.get(group_key)
        if previous is None:
            selected[group_key] = (ts_name, run_item)
        elif ts_name > previous[0]:
            selected[group_key] = (ts_name, run_item)
            skipped_older += 1
        else:
            skipped_older += 1

    discovered = sorted((item for _, item in selected.values()), key=lambda item: item.rel_path)
    return discovered, skipped_non_rubric, skipped_older


def _append_change_log(change_log_path: Path, record: dict[str, Any]) -> None:
    change_log_path.parent.mkdir(parents=True, exist_ok=True)
    with change_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# RoboDojo VLM Rubric Batch Evaluation Summary",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Eval root: `{payload['eval_root']}`",
        f"- Judge model: `{payload['judge_model']}`",
        f"- Update `_result.json`: `{payload['update_result']}` (backup: `{payload['create_backup']}`)",
        f"- Only sim_success: `{payload['only_sim_success']}`",
        f"- Elapsed: `{format_duration(payload.get('elapsed_s', 0.0))}`",
        "",
        "| Task | Status | P | F | U | Skip | ERR | Sim SR -> Final SR | Overturned | Result Update | Run |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for row in payload.get("runs", []):
        counts = row.get("counts", {})
        update = row.get("result_update", {})
        sim_sr = row.get("sim_success_rate")
        final_sr = update.get("success_rate")
        if isinstance(sim_sr, int | float) and isinstance(final_sr, int | float):
            sr_cell = f"`{sim_sr:.3f} -> {final_sr:.3f}`"
        elif isinstance(sim_sr, int | float):
            sr_cell = f"`{sim_sr:.3f}`"
        else:
            sr_cell = "-"
        lines.append(
            f"| `{row.get('task', '')}` | `{row.get('status', '')}` | "
            f"{counts.get('pass', '-')} | {counts.get('fail', '-')} | {counts.get('uncertain', '-')} | "
            f"{counts.get('skipped', '-')} | {counts.get('error', '-')} | "
            f"{sr_cell} | {update.get('overturned_count', '-')} | "
            f"`{update.get('status', '-')}` | `{row.get('rel_path', '')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate RoboDojo eval_result runs against VLM Rubrics and safely manage _result.json updates."
        )
    )
    parser.add_argument(
        "--list-rubrics",
        action="store_true",
        help="List all tasks with built-in VLM rubrics and their criterion IDs, then exit",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=REPO_ROOT / "eval_result",
        help="Root directory of RoboDojo evaluation results (default: <repo>/eval_result)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            f"Optional mirror directory for {RUBRIC_REPORT_JSON}/{RUBRIC_REPORT_MD} "
            "(default: write inside each run directory)"
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of all evaluated runs",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=None,
        help="Optional path to write a Markdown summary table of all evaluated runs",
    )
    parser.add_argument(
        "--change-log",
        type=Path,
        default=None,
        help="Optional JSONL path recording every _result.json modification",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Task name to include (repeatable). Defaults to all Rubric-configured tasks.",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Only evaluate runs whose relative path under --eval-root contains this substring.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Evaluate all historical timestamp directories instead of only the latest timestamp per "
            "(task, policy, embodiment, seed) (default: latest only, matching summarize_result.py)"
        ),
    )
    parser.add_argument(
        "--skip-graded",
        "--skip-already-graded",
        dest="skip_graded",
        action="store_true",
        help="Skip runs whose _result.json is already cleanly graded by VLM Rubric (0 missing, 0 judge_error).",
    )
    parser.add_argument(
        "--dry-run",
        "--status",
        dest="dry_run",
        action="store_true",
        help="Inspect matching runs, grading status, and backups without calling the VLM Judge API or modifying files",
    )
    add_judge_api_args(parser, default_retries=5)
    add_eval_args(parser, default_workers=4, include_max_episodes=True, resume_default=True)
    add_logging_args(parser)

    args = parser.parse_args(argv)
    if args.list_rubrics:
        print(format_rubric_catalog())
        return 0

    validate_common_args(parser, args)

    selected_tasks = set(args.task) if args.task else None
    if selected_tasks:
        unknown = sorted(t for t in selected_tasks if not get_rubric_criteria_prompt(t))
        if unknown:
            parser.error(f"Task(s) have no built-in VLM rubric configured: {', '.join(unknown)}")

    logger, log_path = configure_rubric_logging(
        log_file=args.log_file,
        log_dir=args.log_dir,
        run_name="rubric_status" if args.dry_run else "rubric_eval",
        level=args.log_level,
        quiet=args.quiet,
    )
    if log_path is not None:
        logger.info("log file: %s", log_path)

    latest_only = not args.all_runs
    try:
        runs, skipped_non_rubric, skipped_older = discover_eval_runs(
            args.eval_root,
            tasks=selected_tasks,
            match=args.match,
            latest_only=latest_only,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    resolved_root = args.eval_root.expanduser().resolve()
    logger.info(
        "[rubric-eval] root=%s mode=%s matched_runs=%d skipped_older=%d skipped_non_rubric=%d",
        resolved_root,
        "latest_only" if latest_only else "all_runs",
        len(runs),
        skipped_older,
        skipped_non_rubric,
    )
    if not runs:
        return 0

    if args.dry_run:
        total_eps = 0
        total_sim_succ = 0
        graded_count = 0
        backup_count = 0
        for idx, item in enumerate(runs, 1):
            flags: list[str] = []
            if item.already_graded:
                flags.append("graded")
                graded_count += 1
            if item.has_backup:
                flags.append("backup")
                backup_count += 1
            if not item.video_count:
                flags.append("NO_VIDEOS")
            sr_text = ""
            if isinstance(item.sim_success_rate, int | float) and isinstance(item.current_success_rate, int | float):
                if item.already_graded:
                    sr_text = f" sr={item.sim_success_rate:.3f}->{item.current_success_rate:.3f}"
                else:
                    sr_text = f" sr={item.sim_success_rate:.3f}"
            flag_str = f" [{','.join(flags)}]" if flags else ""
            logger.info(
                "  [%3d/%d] %-28s eps=%3d sim_succ=%3d videos=%3d%s | %s%s",
                idx,
                len(runs),
                item.task,
                item.total_episodes,
                item.sim_success_episodes,
                item.video_count,
                sr_text,
                item.rel_path,
                flag_str,
            )
            total_eps += item.total_episodes
            total_sim_succ += item.sim_success_episodes
        logger.info(
            "[rubric-eval] status total: runs=%d graded=%d backups=%d episodes=%d sim_success=%d",
            len(runs),
            graded_count,
            backup_count,
            total_eps,
            total_sim_succ,
        )
        return 0

    client = build_judge_client(args, client_cls=VLMJudgeClient)

    batch_started = time.monotonic()
    summary_rows: list[dict[str, Any]] = []
    failures = 0
    only_sim_success = args.only_sim_success
    resume = not args.no_resume
    update_result = not args.no_update_result
    create_backup = not args.no_backup

    change_log_path = args.change_log
    if change_log_path is None and args.log_dir is not None:
        change_log_path = Path(args.log_dir).expanduser().resolve() / "result_changes.jsonl"

    for idx, item in enumerate(runs, 1):
        if args.skip_graded and item.already_graded:
            logger.info("[%d/%d] SKIP (already graded) %s", idx, len(runs), item.rel_path)
            summary_rows.append(
                {
                    "task": item.task,
                    "run_dir": str(item.run_dir),
                    "rel_path": item.rel_path,
                    "status": "skipped_already_graded",
                    "sim_success_rate": item.sim_success_rate,
                }
            )
            continue
        if item.video_count == 0:
            logger.info("[%d/%d] SKIP (no episode videos) %s", idx, len(runs), item.rel_path)
            summary_rows.append(
                {
                    "task": item.task,
                    "run_dir": str(item.run_dir),
                    "rel_path": item.rel_path,
                    "status": "skipped_no_videos",
                    "sim_success_rate": item.sim_success_rate,
                }
            )
            continue

        out_dir = (args.output_root / item.rel_path) if args.output_root else None
        logger.info(
            "[%d/%d] === %s | sim_succ=%d/%d | %s ===",
            idx,
            len(runs),
            item.task,
            item.sim_success_episodes,
            item.total_episodes,
            item.rel_path,
        )
        report = evaluate_run(
            item.run_dir,
            output_dir=out_dir,
            task=item.task,
            sample_fraction=args.sample_fraction,
            max_episodes=args.max_episodes,
            view_policy=args.view_policy,
            update_result=update_result,
            create_backup=create_backup,
            skip_update_on_error=args.skip_update_on_error,
            workers=args.workers,
            client=client,
            logger=logger,
            resume=resume,
            only_sim_success=only_sim_success,
        )
        counts = summarize_episodes(report.get("episodes", {}))
        update_info = report.get("result_update", {})
        update_status = update_info.get("status", "unknown")
        if counts["error"] > 0 or update_status == "error":
            failures += 1

        sim_sr = item.sim_success_rate if item.sim_success_rate is not None else 0.0
        final_sr = update_info.get("success_rate")
        sr_note = f"{sim_sr:.3f}->{final_sr:.3f}" if isinstance(final_sr, int | float) else f"{sim_sr:.3f}"
        logger.info(
            "[rubric-eval] done %s | P=%d F=%d U=%d Skip=%d ERR=%d | result=%s sr=%s elapsed=%s",
            item.task,
            counts["pass"],
            counts["fail"],
            counts["uncertain"],
            counts["skipped"],
            counts["error"],
            update_status,
            sr_note,
            format_duration(report.get("elapsed_s", 0.0)),
        )
        if update_status == "updated" and change_log_path is not None:
            _append_change_log(
                change_log_path,
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "task": item.task,
                    "run_dir": str(item.run_dir),
                    "rel_path": item.rel_path,
                    "result_path": update_info.get("path"),
                    "sim_backup_path": update_info.get("sim_backup_path"),
                    "sim_success_rate": update_info.get("sim_success_rate"),
                    "final_success_rate": update_info.get("success_rate"),
                    "sim_score": update_info.get("sim_score"),
                    "final_score": update_info.get("score"),
                    "overturned_count": update_info.get("overturned_count", 0),
                    "overturned_episodes": update_info.get("overturned_episodes", []),
                },
            )
        summary_rows.append(
            {
                "task": item.task,
                "run_dir": str(item.run_dir),
                "rel_path": item.rel_path,
                "report_dir": report.get("report_dir"),
                "status": "graded",
                "counts": counts,
                "sim_success_rate": item.sim_success_rate,
                "result_update": update_info,
                "elapsed_s": report.get("elapsed_s"),
            }
        )

    batch_elapsed = round(time.monotonic() - batch_started, 2)
    summary_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_root": str(resolved_root),
        "selection_mode": "latest_only" if latest_only else "all_runs",
        "skipped_older": skipped_older,
        "judge_model": client.model,
        "update_result": update_result,
        "create_backup": create_backup,
        "only_sim_success": only_sim_success,
        "elapsed_s": batch_elapsed,
        "log_file": str(log_path) if log_path else None,
        "runs": summary_rows,
    }

    summary_path = args.summary_json
    if summary_path is None and args.output_root is not None:
        summary_path = args.output_root / "rubric_eval_summary.json"
    if summary_path is not None:
        atomic_write_json(summary_path.expanduser().resolve(), summary_payload)
        logger.info("[rubric-eval] wrote JSON summary to %s", summary_path)

    summary_md_path = args.summary_md
    if summary_md_path is None and summary_path is not None:
        summary_md_path = summary_path.with_suffix(".md")
    if summary_md_path is not None:
        _write_summary_markdown(summary_md_path.expanduser().resolve(), summary_payload)
        logger.info("[rubric-eval] wrote Markdown summary to %s", summary_md_path)

    logger.info(
        "[rubric-eval] finished runs=%d failures=%d elapsed=%s",
        len(runs),
        failures,
        format_duration(batch_elapsed),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

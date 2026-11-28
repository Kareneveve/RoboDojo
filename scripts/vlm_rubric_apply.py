#!/usr/bin/env python3
"""Apply VLM Rubric verdicts to `_result.json`, or restore original simulator `_result.json` files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm_rubric import (
    RUBRIC_REPORT_JSON,
    ResultStoreError,
    apply_rubric_verdict,
    configure_rubric_logging,
    restore_sim_result,
)
from src.vlm_rubric.cli_common import add_logging_args

SEED_RE = re.compile(r"^(\d+)_")


def _iter_result_run_dirs(eval_root: Path, *, match: str | None = None, latest_only: bool = False) -> list[Path]:
    root = eval_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {eval_root}")
    candidates: list[tuple[tuple[object, ...], str, Path]] = []
    for result_file in sorted(root.rglob("_result.json")):
        run_dir = result_file.parent
        rel = run_dir.relative_to(root)
        rel_str = rel.as_posix()
        if match and match not in rel_str:
            continue
        parts = rel.parts
        if len(parts) >= 5:
            seed_match = SEED_RE.match(parts[-2])
            seed_key: int | str = int(seed_match.group(1)) if seed_match else parts[-2]
            group_key: tuple[object, ...] = (*parts[:-2], seed_key)
        else:
            group_key = (parts[:-1],)
        candidates.append((group_key, run_dir.name, run_dir))

    if not latest_only:
        return [run_dir for _, _, run_dir in candidates]

    selected: dict[tuple[object, ...], tuple[str, Path]] = {}
    for group_key, ts_name, run_dir in candidates:
        prev = selected.get(group_key)
        if prev is None or ts_name > prev[0]:
            selected[group_key] = (ts_name, run_dir)
    return sorted((run_dir for _, run_dir in selected.values()), key=lambda p: p.as_posix())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Apply a VLM Rubric report ({RUBRIC_REPORT_JSON}) into _result.json, "
            "or restore original simulator _result.json."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", type=Path, help="Single run directory containing _result.json")
    target.add_argument(
        "--eval-root",
        type=Path,
        help="Batch root directory to apply or restore all matching runs containing _result.json",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="With --eval-root, only touch runs whose relative path contains this substring",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="With --eval-root in apply mode, include older timestamp folders instead of latest only",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"Rubric evaluation JSON for single-run apply (default: <run-dir>/{RUBRIC_REPORT_JSON})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional alternate output path for graded result JSON (single --run-dir only)",
    )
    parser.add_argument(
        "--allow-partial-views",
        action="store_true",
        help="Allow rubric pass with missing cameras. The default requires three_view coverage.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating _result.sim_backup.json before modifying _result.json in-place",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore _result.json back to original simulator values (prefers _result.sim_backup.json)",
    )
    parser.add_argument(
        "--remove-backup",
        action="store_true",
        help="With --restore, remove _result.sim_backup.json after restoring _result.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the apply or restore without modifying files on disk",
    )
    add_logging_args(parser)

    args = parser.parse_args(argv)
    if args.eval_root and (args.report or args.output):
        parser.error("--report and --output can only be used with --run-dir")

    mode = "restore" if args.restore else "apply"
    logger, log_path = configure_rubric_logging(
        log_file=args.log_file,
        log_dir=args.log_dir,
        run_name=f"result_{mode}",
        level=args.log_level,
        quiet=args.quiet,
    )
    if log_path is not None:
        logger.info("log file: %s", log_path)

    if args.run_dir:
        run_dirs = [args.run_dir.expanduser().resolve()]
    else:
        try:
            # In apply mode default to latest timestamp per (task, policy, embodiment, seed);
            # in restore mode scan all timestamp folders so no previously modified run is missed.
            run_dirs = _iter_result_run_dirs(
                args.eval_root,
                match=args.match,
                latest_only=(not args.restore and not args.all_runs),
            )
        except FileNotFoundError as exc:
            parser.error(str(exc))

    if args.restore:
        restored = 0
        unchanged = 0
        errors = 0
        for run_dir in run_dirs:
            try:
                info = restore_sim_result(
                    run_dir,
                    write=not args.dry_run,
                    remove_backup=args.remove_backup,
                )
            except ResultStoreError as exc:
                errors += 1
                logger.error("restore failed for %s: %s", run_dir, exc)
                continue
            if info["status"] == "restored":
                restored += 1
                prefix = "[dry-run] would restore" if args.dry_run else "restored"
                logger.info(
                    "%s %s (from=%s sr=%s score=%s)",
                    prefix,
                    info["path"],
                    info["restored_from"],
                    info.get("success_rate"),
                    info.get("score"),
                )
            else:
                unchanged += 1
                if args.run_dir:
                    logger.info("already clean: %s", info["path"])
        logger.info(
            "restore summary: total=%d restored=%d unchanged=%d errors=%d",
            len(run_dirs),
            restored,
            unchanged,
            errors,
        )
        return 1 if errors else 0

    updated = 0
    skipped = 0
    errors = 0
    for run_dir in run_dirs:
        if args.run_dir:
            report_file = args.report
        else:
            report_file = run_dir / RUBRIC_REPORT_JSON
            if not report_file.exists() and (run_dir / "vlm_review.json").exists():
                report_file = run_dir / "vlm_review.json"
            if not report_file.exists():
                skipped += 1
                continue
        try:
            result = apply_rubric_verdict(
                run_dir,
                report_path=report_file,
                output_path=args.output if args.run_dir else None,
                require_three_view=not args.allow_partial_views,
                write=not args.dry_run,
                create_backup=not args.no_backup,
            )
        except ResultStoreError as exc:
            if args.run_dir:
                parser.error(str(exc))
            errors += 1
            logger.error("apply failed for %s: %s", run_dir, exc)
            continue

        updated += 1
        if args.dry_run and args.run_dir:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue

        target_path = args.output if (args.run_dir and args.output) else (run_dir / "_result.json")
        changes = result.get("rubric_eval", {}).get("changes", {})
        prefix = "[dry-run] would update" if args.dry_run else "updated"
        logger.info(
            "%s %s | sr=%s->%.4f score=%s->%.2f overturned=%d",
            prefix,
            target_path,
            changes.get("sim_success_rate"),
            result.get("success_rate", 0.0),
            changes.get("sim_score"),
            result.get("score", 0.0),
            changes.get("overturned_count", 0),
        )

    if args.eval_root:
        logger.info(
            "apply summary: total=%d updated=%d skipped=%d errors=%d",
            len(run_dirs),
            updated,
            skipped,
            errors,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

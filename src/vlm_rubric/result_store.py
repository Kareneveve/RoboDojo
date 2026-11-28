"""Apply, backup, and restore RoboDojo `_result.json` files for VLM Rubric evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

SIM_BACKUP_FILENAME = "_result.sim_backup.json"
RUBRIC_REPORT_JSON = "rubric_eval.json"
RUBRIC_REPORT_MD = "rubric_eval.md"

_TOP_LEVEL_RUBRIC_KEYS = {
    "sim_success_rate",
    "sim_score",
    "sim_eval_time",
    "rubric_eval",
    "vlm_adjudication",
    "result_source",
    "result_version",
}

_DETAIL_RUBRIC_KEYS = {
    "sim_success",
    "sim_score",
    "rubric_verdict",
    "rubric_confidence",
    "rubric_judge_status",
    "rubric_camera_mode",
    "rubric_missing_views",
    "rubric_violation_detected",
    "rubric_eval_path",
    "final_status",
    "final_success",
    "final_score",
    "final_decision_reason",
    # Legacy keys stripped cleanly on restore
    "vlm_verdict",
    "vlm_confidence",
    "vlm_review_status",
    "vlm_camera_mode",
    "vlm_missing_views",
    "vlm_shortcut_detected",
    "vlm_review_path",
}


class ResultStoreError(RuntimeError):
    """Raised when a VLM Rubric evaluation report cannot be applied to or restored from `_result.json`."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultStoreError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResultStoreError(f"JSON root must be an object: {path}")
    return value


def _episode_key(value: object) -> str | None:
    text = str(value)
    if text.lstrip("-").isdigit():
        return str(int(text))
    return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a UTF-8 JSON object with fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def is_rubric_graded_result(payload: dict[str, Any]) -> bool:
    """Return True if a `_result.json` payload has already been updated with VLM Rubric verdicts."""
    return (
        payload.get("result_source") in {"simulator_plus_vlm_rubric", "simulator_plus_vlm_posthoc"}
        or "rubric_eval" in payload
        or "vlm_adjudication" in payload
    )


def backup_sim_result(result_file: Path, *, backup_path: Path | None = None) -> Path | None:
    """Create a one-time verbatim backup of an untouched simulator `_result.json`.

    If the backup already exists it is never overwritten, preserving the first
    pristine simulator snapshot across repeated rubric re-evaluations.
    """
    target_backup = backup_path or (result_file.parent / SIM_BACKUP_FILENAME)
    if target_backup.exists():
        return target_backup
    if not result_file.exists():
        return None
    raw = _read_json(result_file)
    if is_rubric_graded_result(raw):
        return None
    atomic_write_json(target_backup, raw)
    return target_backup


def _episode_judge_status(entry: dict[str, Any]) -> str:
    return str(entry.get("judge_status", entry.get("review_status", "error")))


def _episode_verdict(entry: dict[str, Any]) -> str:
    return str(entry.get("verdict", entry.get("vlm_verdict", "uncertain"))).lower()


def _episode_violation(entry: dict[str, Any]) -> bool:
    return bool(entry.get("violation_detected", entry.get("shortcut_detected", False)))


def _final_decision(
    detail: dict[str, Any],
    judgement: dict[str, Any] | None,
    *,
    require_three_view: bool,
) -> tuple[str, bool, float, str]:
    """Return final status, success, score, and a machine-readable reason."""

    sim_success = bool(detail.get("sim_success", detail.get("success", False)))
    sim_score = float(detail.get("sim_score", detail.get("score", 0.0)) or 0.0)
    if not sim_success:
        return "fail", False, 0.0, "simulator_not_success"
    if judgement is None:
        return "uncertain", False, 0.0, "missing_rubric_eval"
    if _episode_judge_status(judgement) != "completed":
        return "uncertain", False, 0.0, "rubric_judge_error"
    if require_three_view and judgement.get("camera_mode") != "three_view":
        return "uncertain", False, 0.0, "required_three_view_missing"
    verdict = _episode_verdict(judgement)
    if verdict == "uncertain":
        return "uncertain", False, 0.0, "rubric_uncertain"
    if verdict != "pass":
        return "fail", False, 0.0, "rubric_not_pass"
    return "pass", True, sim_score, "simulator_and_rubric_pass"


def apply_rubric_verdict(
    run_dir: str | Path,
    *,
    report_path: str | Path | None = None,
    result_path: str | Path | None = None,
    output_path: str | Path | None = None,
    require_three_view: bool = True,
    write: bool = True,
    create_backup: bool = True,
) -> dict[str, Any]:
    """Apply a completed VLM Rubric report (`rubric_eval.json`) into `_result.json`.

    The original simulator values are preserved under `sim_success` / `sim_score`
    before the canonical `success` / `score` fields are updated. When mutating
    `_result.json` in-place with `create_backup=True` (default), a one-time
    verbatim snapshot is first written to `_result.sim_backup.json`.
    """

    run_path = Path(run_dir).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve() if result_path else run_path / "_result.json"
    if report_path is not None:
        report_file = Path(report_path).expanduser().resolve()
    else:
        report_file = run_path / RUBRIC_REPORT_JSON
        if not report_file.exists() and (run_path / "vlm_review.json").exists():
            report_file = run_path / "vlm_review.json"
    target_file = Path(output_path).expanduser().resolve() if output_path else result_file

    result = _read_json(result_file)
    report = _read_json(report_file)
    details = result.get("details")
    episodes = report.get("episodes")
    if not isinstance(details, dict):
        raise ResultStoreError(f"Result has no details object: {result_file}")
    if not isinstance(episodes, dict):
        raise ResultStoreError(f"Rubric report has no episodes object: {report_file}")

    normalized_details = {
        key: value
        for raw_key, value in details.items()
        if (key := _episode_key(raw_key)) is not None and isinstance(value, dict)
    }
    normalized_judgements = {
        key: value
        for raw_key, value in episodes.items()
        if (key := _episode_key(raw_key)) is not None and isinstance(value, dict)
    }
    if not normalized_details:
        raise ResultStoreError(f"Result details are empty: {result_file}")

    backup_file: Path | None = None
    in_place = target_file == result_file
    if write and in_place and create_backup:
        backup_file = backup_sim_result(result_file)
    elif (result_file.parent / SIM_BACKUP_FILENAME).exists():
        backup_file = result_file.parent / SIM_BACKUP_FILENAME

    generated_at = datetime.now(UTC).isoformat()
    counts = {"pass": 0, "fail": 0, "uncertain": 0, "skipped": 0, "missing": 0, "judge_error": 0}
    status_counts = {"pass": 0, "fail": 0, "uncertain": 0}
    overturned_episodes: list[dict[str, Any]] = []

    for key, detail in normalized_details.items():
        if "sim_success" not in detail:
            detail["sim_success"] = bool(detail.get("success", False))
        if "sim_score" not in detail:
            detail["sim_score"] = float(detail.get("score", 0.0) or 0.0)

        judgement = normalized_judgements.get(key)
        if judgement is None:
            counts["missing"] += 1
        elif _episode_judge_status(judgement) == "skipped":
            counts["skipped"] += 1
        elif _episode_judge_status(judgement) != "completed":
            counts["judge_error"] += 1
        else:
            verdict = _episode_verdict(judgement)
            counts[verdict if verdict in {"pass", "fail", "uncertain"} else "uncertain"] += 1

        final_status, final_success, final_score, reason = _final_decision(
            detail,
            judgement,
            require_three_view=require_three_view,
        )
        if detail["sim_success"] and not final_success:
            overturned_episodes.append(
                {
                    "episode_id": key,
                    "reason": reason,
                    "verdict": _episode_verdict(judgement) if judgement else None,
                    "sim_score": detail["sim_score"],
                }
            )

        detail["rubric_verdict"] = _episode_verdict(judgement) if judgement else None
        detail["rubric_confidence"] = judgement.get("confidence") if judgement else None
        detail["rubric_judge_status"] = _episode_judge_status(judgement) if judgement else "missing"
        detail["rubric_camera_mode"] = judgement.get("camera_mode") if judgement else None
        detail["rubric_missing_views"] = judgement.get("missing_views", []) if judgement else []
        detail["rubric_violation_detected"] = _episode_violation(judgement) if judgement else False
        detail["rubric_eval_path"] = str(report_file)
        detail["final_status"] = final_status
        detail["final_success"] = final_success
        detail["final_score"] = final_score
        detail["final_decision_reason"] = reason
        detail["success"] = final_success
        detail["score"] = final_score
        status_counts[final_status] += 1

    total = len(normalized_details)
    final_successes = sum(bool(detail.get("final_success")) for detail in normalized_details.values())
    final_total_score = sum(float(detail.get("final_score", 0.0) or 0.0) for detail in normalized_details.values())
    final_success_rate = final_successes / total if total else 0.0
    final_score_pct = final_total_score / total * 100 if total else 0.0

    original_success_rate = result.get("sim_success_rate", result.get("success_rate"))
    original_score = result.get("sim_score", result.get("score"))
    original_eval_time = result.get("sim_eval_time", result.get("eval_time"))

    sim_sr_num = float(original_success_rate) if isinstance(original_success_rate, int | float) else None
    sim_score_num = float(original_score) if isinstance(original_score, int | float) else None
    changes = {
        "sim_success_rate": sim_sr_num,
        "final_success_rate": final_success_rate,
        "success_rate_delta": round(final_success_rate - sim_sr_num, 6) if sim_sr_num is not None else None,
        "sim_score": sim_score_num,
        "final_score": final_score_pct,
        "score_delta": round(final_score_pct - sim_score_num, 4) if sim_score_num is not None else None,
        "overturned_count": len(overturned_episodes),
        "overturned_episodes": overturned_episodes,
    }

    result["sim_success_rate"] = original_success_rate
    result["sim_score"] = original_score
    if "eval_time" in result:
        result["sim_eval_time"] = original_eval_time
    rubric_summary: dict[str, Any] = {
        "version": "robodojo-vlm-rubric-eval/v1",
        "generated_at": generated_at,
        "report_path": str(report_file),
        "require_three_view": require_three_view,
        "counts": counts,
        "final_status_counts": status_counts,
        "episodes": total,
        "changes": changes,
    }
    if backup_file is not None:
        rubric_summary["sim_backup_path"] = str(backup_file)
    result.pop("vlm_adjudication", None)
    result["rubric_eval"] = rubric_summary
    result["success_rate"] = final_success_rate
    result["score"] = final_score_pct
    if "eval_time" in result:
        result["eval_time"] = original_eval_time
    result["result_source"] = "simulator_plus_vlm_rubric"
    result["result_version"] = "robodojo-result/vlm-rubric-v1"
    result["details"] = normalized_details

    if write:
        atomic_write_json(target_file, result)
    return result


def restore_sim_result(
    run_dir: str | Path,
    *,
    result_path: str | Path | None = None,
    backup_path: str | Path | None = None,
    write: bool = True,
    remove_backup: bool = False,
) -> dict[str, Any]:
    """Restore `_result.json` back to the untouched simulator evaluation result.

    Prefers the verbatim `_result.sim_backup.json` snapshot when present. If no
    backup file exists, reconstructs the simulator payload from `sim_*` fields
    and strips all rubric / legacy VLM metadata.
    """

    run_path = Path(run_dir).expanduser().resolve()
    result_file = Path(result_path).expanduser().resolve() if result_path else run_path / "_result.json"
    backup_file = Path(backup_path).expanduser().resolve() if backup_path else result_file.parent / SIM_BACKUP_FILENAME

    if backup_file.exists():
        restored = _read_json(backup_file)
        if write:
            atomic_write_json(result_file, restored)
            if remove_backup:
                try:
                    backup_file.unlink()
                except OSError:
                    pass
        return {
            "status": "restored",
            "restored_from": "backup",
            "path": str(result_file),
            "backup_path": str(backup_file),
            "success_rate": restored.get("success_rate"),
            "score": restored.get("score"),
        }

    current = _read_json(result_file)
    if not is_rubric_graded_result(current):
        return {
            "status": "unchanged",
            "restored_from": "already_clean",
            "path": str(result_file),
            "success_rate": current.get("success_rate"),
            "score": current.get("score"),
        }

    reconstructed: dict[str, Any] = {k: v for k, v in current.items() if k not in _TOP_LEVEL_RUBRIC_KEYS}
    if "sim_success_rate" in current and current["sim_success_rate"] is not None:
        reconstructed["success_rate"] = current["sim_success_rate"]
    if "sim_score" in current and current["sim_score"] is not None:
        reconstructed["score"] = current["sim_score"]
    if "sim_eval_time" in current:
        reconstructed["eval_time"] = current["sim_eval_time"]

    details = current.get("details")
    if isinstance(details, dict):
        clean_details: dict[str, Any] = {}
        for key, detail in details.items():
            if not isinstance(detail, dict):
                clean_details[key] = detail
                continue
            cleaned = {k: v for k, v in detail.items() if k not in _DETAIL_RUBRIC_KEYS}
            if "sim_success" in detail:
                cleaned["success"] = bool(detail["sim_success"])
            if "sim_score" in detail:
                cleaned["score"] = float(detail["sim_score"])
            clean_details[key] = cleaned
        reconstructed["details"] = clean_details

    if write:
        atomic_write_json(result_file, reconstructed)
    return {
        "status": "restored",
        "restored_from": "reconstructed",
        "path": str(result_file),
        "success_rate": reconstructed.get("success_rate"),
        "score": reconstructed.get("score"),
    }


__all__ = [
    "RUBRIC_REPORT_JSON",
    "RUBRIC_REPORT_MD",
    "ResultStoreError",
    "SIM_BACKUP_FILENAME",
    "apply_rubric_verdict",
    "atomic_write_json",
    "backup_sim_result",
    "is_rubric_graded_result",
    "restore_sim_result",
]

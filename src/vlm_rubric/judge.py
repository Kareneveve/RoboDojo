"""Run a synchronized multi-camera VLM-as-a-Judge rubric evaluation over a RoboDojo run."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

from .client import VLMJudgeClient
from .logging_utils import format_duration
from .media import (
    CAMERAS,
    EpisodeMedia,
    MediaError,
    discover_episodes,
    image_data_url,
    load_result,
    stitch_camera_frames,
    synchronized_frames,
)
from .result_store import (
    RUBRIC_REPORT_JSON,
    RUBRIC_REPORT_MD,
    ResultStoreError,
    apply_rubric_verdict,
    atomic_write_json,
)
from .rubrics import get_rubric_criteria_prompt

SYSTEM_PROMPT = (
    "You are a strict robotics benchmark rubric judge. You receive synchronized temporal samples from up to three "
    "camera views of one episode; when multiple views are present, each timestamp is stitched into one horizontal "
    "composite image ordered left-to-right with each camera labeled at the top of its panel. Use all views together "
    "and do not infer a hidden action from one camera when another view contradicts it. The simulator score and final "
    "arrangement do not prove that the required method was used. Treat the task-specific rubric criteria as the "
    "primary grading contract. For every rubric criterion, inspect the visible action sequence across early, middle, "
    "and late timestamps, allowed tool or gripper contact, and method-relevant object support. Explicitly mark a "
    "violation when the result is achieved with a forbidden tool, a direct push, a sweep, a throw, a collision, or a "
    "hidden pre-positioned object. Distinguish grasping or pushing an allowed tool from directly touching the target "
    "object: when a rubric requires using a tool, gripper contact with the tool itself is expected, and only direct "
    "gripper contact with forbidden target objects is a violation. Do not treat a target object visible near the edge "
    "or background of a wrist camera as gripper contact unless head and wrist views jointly confirm actual "
    "finger-to-object contact. Similarly, releasing the tool and retracting the arms in late frames after the task "
    "action has occurred in earlier or middle frames is normal completion behavior and must not be cited as missing "
    "tool contact. Use wrist views to verify gripper contact and release, and use the head view to identify the "
    "required tool and its contact with the object. Ignore numeric thresholds, exact positions, object counts, final "
    "geometry, reset poses, and other outcomes owned by the simulator evaluator. Do not mark a rubric criterion as "
    "met from a single initial or final frame alone. Return one entry in `criteria` for each required criterion ID "
    "and cite synchronized frame/time and camera evidence for your judgment. Return only strictly valid JSON matching "
    "the requested schema. Keep the response compact with at most 4 criteria, 6 evidence items, and 8 temporal "
    "events. Keep descriptions under 25 words. If the evidence is not clear, use `uncertain` rather than guessing."
)

AVAILABLE_VIEW_PROMPT = (
    "The episode does not contain all required cameras. Evaluate only the supplied views, treat the evidence as "
    "view-limited, and return `uncertain` when the missing cameras could change the judgment."
)


def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                value = None
        else:
            value = None
        if value is None:
            value = _repair_truncated_json(cleaned)
    if not isinstance(value, dict):
        raise ValueError("VLM Judge response JSON must be an object")
    return value


def _repair_truncated_json(text: str) -> Any:
    """Recover a JSON object cut off after a complete nested value."""

    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    candidate_text = text[start:]
    for position in range(len(candidate_text) - 1, -1, -1):
        if candidate_text[position] != "}":
            continue
        fragment = candidate_text[: position + 1].rstrip()
        while fragment.endswith(","):
            fragment = fragment[:-1].rstrip()
        stack: list[str] = []
        in_string = False
        escaped = False
        valid = True
        for char in fragment:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                if not stack or (char == "]" and stack[-1] != "[") or (char == "}" and stack[-1] != "{"):
                    valid = False
                    break
                stack.pop()
        if not valid or in_string:
            continue
        suffix = "".join("]" if char == "[" else "}" for char in reversed(stack))
        try:
            return json.loads(fragment + suffix)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("Could not repair truncated JSON", text, len(text))


def _normalize_criterion(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    normalized = dict(item)
    if "met" not in normalized and "pass" in normalized:
        normalized["met"] = bool(normalized.pop("pass"))
    elif "met" in normalized:
        normalized["met"] = bool(normalized["met"])
    return normalized


def _normalize_judge_output(value: dict[str, Any]) -> dict[str, Any]:
    verdict = str(value.get("verdict", value.get("vlm_verdict", "uncertain"))).lower()
    if verdict not in {"pass", "fail", "uncertain"}:
        verdict = "uncertain"
    try:
        confidence = min(1.0, max(0.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    raw_criteria = value.get("criteria", value.get("checks", []))
    criteria = (
        [c for item in raw_criteria if (c := _normalize_criterion(item)) is not None]
        if isinstance(raw_criteria, list)
        else []
    )
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    temporal_events = value.get("temporal_events", [])
    if not isinstance(temporal_events, list):
        temporal_events = [str(temporal_events)]
    violation_flag = value.get("violation_detected", value.get("shortcut_detected", verdict == "fail"))
    return {
        "verdict": verdict,
        "confidence": confidence,
        "violation_detected": bool(violation_flag),
        "criteria": criteria,
        "evidence": evidence,
        "temporal_events": temporal_events,
        "camera_assessment": value.get("camera_assessment", {}),
    }


def extract_criterion_ids(rubric: str) -> list[str]:
    """Extract the named rubric criterion IDs from a task rubric specification."""

    ids: list[str] = []
    pattern = (
        r"(?:Rubric criteria|Visual audit checks|视觉审查检查项)"
        r"(?:\s*\[([^\]]+)\])?[:：]|^\[([A-Za-z][A-Za-z0-9_-]*)\]"
    )
    for header_ids, line_id in re.findall(pattern, rubric, flags=re.MULTILINE):
        if header_ids:
            ids.extend(item.strip() for item in header_ids.split(",") if item.strip())
        if line_id:
            ids.append(line_id)
    return ids


def _prompt_for_episode(
    task: str,
    media: EpisodeMedia,
    timestamps: list[float],
    cameras: tuple[str, ...],
    rubric_override: str | None = None,
) -> str:
    builtin_rubric = get_rubric_criteria_prompt(task)
    if rubric_override:
        rubric = "\n\n".join(part for part in (builtin_rubric, rubric_override) if part.strip())
    else:
        rubric = builtin_rubric
    criterion_ids = extract_criterion_ids(rubric)
    missing = sorted(set(CAMERAS) - set(cameras))
    view_description = ", ".join(cameras)
    if len(cameras) > 1:
        frame_layout_note = (
            f"Each image below is a synchronized horizontal {len(cameras)}-panel composite at one timestamp, "
            f"ordered left-to-right as {view_description} with each camera labeled on its top bar. "
            "Images are ordered chronologically."
        )
    else:
        frame_layout_note = f"Each image below is one {view_description} frame, ordered chronologically by time."
    return f"""Grade RoboDojo episode {media.episode_id} against the task rubric using synchronized views.

Task: {task or "unknown"}
Automated simulator outcome is metadata only and must not affect this rubric judgment.
Available cameras: {view_description}
Missing cameras: {", ".join(missing) if missing else "none"}
{frame_layout_note}
Sample times in seconds: {json.dumps([round(value, 3) for value in timestamps])}

Task rubric criteria:
{rubric}

Evaluate each named criterion independently. A passing final arrangement cannot override an unmet method
or sequence criterion.
Required criterion IDs: {json.dumps(criterion_ids)}
Scope this evaluation strictly to procedural legitimacy. Ignore numeric thresholds, exact placement or orientation
errors, object counts, final layout, reset poses, and other checks owned by the simulator evaluator. Judge whether the
required tool, contact mode, and action sequence satisfy the rubric and reject shortcuts or policy violations.
For every unmet or uncertain criterion, include synchronized frame/time evidence and the camera that shows it. When a
rubric requires a particular tool, distinguish tool contact from gripper contact. When a rubric requires a grasp,
verify only the visible grasp/lift/transport/release chain needed to establish that method.

Return this JSON shape:
{{
  "verdict": "pass|fail|uncertain",
  "confidence": 0.0,
  "violation_detected": false,
  "criteria": [{{"id": "...", "met": true, "confidence": 0.0,
                 "evidence": [{{"frame_index": 0, "time_s": 0.0,
                                "cameras": ["head"], "description": "..."}}]}}],
  "evidence": [{{"frame_index": 0, "time_s": 0.0,
                 "cameras": {json.dumps(list(cameras))},
                 "description": "..."}}],
  "temporal_events": ["..."],
  "camera_assessment": {{"head": "...", "left_wrist": "...", "right_wrist": "..."}}
}}
"""


def _message_content(
    timestamps: list[float],
    frames: dict[str, list[bytes]],
    cameras: tuple[str, ...] = CAMERAS,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    panel_order = ",".join(cameras)
    for frame_index, timestamp in enumerate(timestamps):
        if len(cameras) > 1:
            composite = stitch_camera_frames(
                {camera: frames[camera][frame_index] for camera in cameras},
                cameras,
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"frame_index={frame_index}, time_s={timestamp:.3f}, "
                        f"panels_left_to_right={panel_order}"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(composite), "detail": "high"},
                }
            )
        else:
            camera = cameras[0]
            content.append(
                {
                    "type": "text",
                    "text": f"frame_index={frame_index}, time_s={timestamp:.3f}, camera={camera}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(frames[camera][frame_index]), "detail": "high"},
                }
            )
    return content


def evaluate_run(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    task: str | None = None,
    model: str | None = None,
    sample_fraction: float = 0.10,
    max_episodes: int | None = None,
    episode_ids: Sequence[str] | set[str] | None = None,
    rubric: str | None = None,
    view_policy: str = "strict",
    update_result: bool = True,
    create_backup: bool = True,
    result_output_path: str | Path | None = None,
    skip_update_on_error: bool = False,
    workers: int = 1,
    client: VLMJudgeClient | None = None,
    progress: Callable[[str], None] | None = print,
    logger: logging.Logger | None = None,
    resume: bool = False,
    only_sim_success: bool = False,
) -> dict[str, Any]:
    """Evaluate every episode in a saved run against its VLM rubric and write JSON/Markdown reports."""

    if view_policy not in {"strict", "available"}:
        raise ValueError(f"Unsupported view policy: {view_policy}")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be greater than 0 and at most 1")
    if workers < 1:
        raise ValueError("workers must be positive")
    frame_stride = max(1, math.ceil(1 / sample_fraction))
    path = Path(run_dir).expanduser().resolve()
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else path
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / RUBRIC_REPORT_JSON

    cached_episodes: dict[str, dict[str, Any]] = {}
    if resume:
        checkpoint_path = report_path if report_path.exists() else (report_dir / "vlm_review.json")
        if checkpoint_path.exists():
            try:
                prev_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if isinstance(prev_payload, dict) and isinstance(prev_payload.get("episodes"), dict):
                    cached_episodes = {
                        str(ep_id): entry
                        for ep_id, entry in prev_payload["episodes"].items()
                        if isinstance(entry, dict)
                        and entry.get("judge_status", entry.get("review_status")) == "completed"
                    }
            except (OSError, json.JSONDecodeError):
                cached_episodes = {}

    result_payload = load_result(path)
    all_episodes = discover_episodes(path, result_payload)
    episodes = all_episodes
    partial_eval = False
    if episode_ids is not None:
        wanted = {str(eid) for eid in episode_ids}
        episodes = [media for media in episodes if str(media.episode_id) in wanted]
        if len(episodes) < len(all_episodes):
            partial_eval = True
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        partial_eval = partial_eval or max_episodes < len(episodes)
        episodes = episodes[:max_episodes]
    if task:
        resolved_task = task
    elif len(path.parents) > 3:
        resolved_task = path.parents[3].name
    else:
        resolved_task = "unknown"
    rubric_configured = rubric is not None or bool(get_rubric_criteria_prompt(resolved_task))
    judge = (client or VLMJudgeClient(model=model)) if rubric_configured else client
    model_name = getattr(judge, "model", None) or model or "not_loaded"
    output: dict[str, Any] = {
        "schema_version": "robodojo-vlm-rubric/v1",
        "run_dir": str(path),
        "task": resolved_task,
        "judge_model": model_name,
        "view_policy": view_policy,
        "sampling": {
            "mode": "uniform_fixed_stride",
            "sample_fraction": sample_fraction,
            "frame_stride": frame_stride,
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "episodes": {},
        "report_dir": str(report_dir),
    }

    progress_lock = threading.Lock()
    save_lock = threading.Lock()
    episode_order = [media.episode_id for media in episodes]
    completed_map: dict[str, dict[str, Any]] = {
        ep_id: cached_episodes[ep_id] for ep_id in episode_order if ep_id in cached_episodes
    }

    def _record_and_flush(ep_id: str, entry_obj: dict[str, Any]) -> None:
        with save_lock:
            completed_map[ep_id] = entry_obj
            if not resume:
                return
            output["episodes"] = {eid: completed_map[eid] for eid in episode_order if eid in completed_map}
            atomic_write_json(report_path, output)

    def _emit(message: str, *, level: int = logging.INFO) -> None:
        if logger is not None:
            logger.log(level, message)
            return
        if progress:
            with progress_lock:
                try:
                    progress(message)
                except OSError:
                    pass

    def _judge_episode(media: EpisodeMedia) -> dict[str, Any]:
        if media.episode_id in cached_episodes:
            cached = cached_episodes[media.episode_id]
            verdict = cached.get("verdict", cached.get("vlm_verdict"))
            _emit(f"[rubric] ep={media.episode_id} cached verdict={verdict} sim={cached.get('sim_success')}")
            return cached
        ep_started = time.monotonic()
        entry: dict[str, Any] = {
            "episode_id": media.episode_id,
            "layout_id": media.result.get("layout_id"),
            "sim_success": bool(media.result.get("sim_success", media.result.get("success", False))),
            "sim_score": media.result.get("sim_score", media.result.get("score", 0.0)),
            "camera_coverage": {camera: camera in media.cameras for camera in CAMERAS},
            "missing_views": sorted(set(CAMERAS) - set(media.cameras)),
        }
        available_cameras = tuple(camera for camera in CAMERAS if camera in media.cameras)
        entry["camera_mode"] = (
            "three_view"
            if not entry["missing_views"]
            else ("head_only" if available_cameras == ("head",) else "partial_views")
        )
        if only_sim_success and not entry["sim_success"]:
            entry.update({"judge_status": "skipped", "error": "sim_success is false"})
            _record_and_flush(media.episode_id, entry)
            return entry
        if not rubric_configured:
            entry.update({"judge_status": "skipped", "error": "no task-specific rubric configured"})
            return entry
        if entry["missing_views"] and view_policy == "strict":
            entry.update({"judge_status": "error", "error": "missing required camera video"})
            missing_views = ", ".join(entry["missing_views"])
            _emit(
                f"[rubric] ep={media.episode_id} error: missing required camera video ({missing_views})",
                level=logging.WARNING,
            )
            return entry
        if not available_cameras:
            entry.update({"judge_status": "error", "error": "no camera video found"})
            _emit(f"[rubric] ep={media.episode_id} error: no camera video found", level=logging.WARNING)
            return entry

        try:
            sampling_metadata: dict[str, Any] = {}
            timestamps, frames = synchronized_frames(
                media,
                sample_fraction,
                cameras=available_cameras,
                metadata=sampling_metadata,
            )
            entry["sampling"] = sampling_metadata or {
                "sample_fraction": sample_fraction,
                "frame_stride": frame_stride,
            }
            entry["sampling"]["frame_layout"] = (
                "horizontal_triptych"
                if len(available_cameras) == 3
                else ("horizontal_composite" if len(available_cameras) > 1 else "single_view")
            )
            entry["sample_count"] = len(timestamps)
            source_counts = sampling_metadata.get("source_frame_counts", {})
            source_frame_count = min(source_counts.values()) if source_counts else "unknown"
            _emit(
                f"[rubric] ep={media.episode_id} start samples={len(timestamps)} "
                f"stride={frame_stride} source_frames={source_frame_count} "
                f"layout={entry['sampling']['frame_layout']}"
            )
            episode_prompt = _prompt_for_episode(
                resolved_task,
                media,
                timestamps,
                available_cameras,
                rubric_override=rubric,
            )
            system_prompt = SYSTEM_PROMPT
            if entry["missing_views"]:
                system_prompt += "\n\n" + AVAILABLE_VIEW_PROMPT
            system_prompt += "\n\n" + episode_prompt
            message_content = _message_content(timestamps, frames, available_cameras)
            raw_response = judge.complete(
                message_content,
                system_prompt=system_prompt,
            )
            try:
                parsed = _normalize_judge_output(_json_from_text(raw_response))
            except (ValueError, json.JSONDecodeError):
                retry_prompt = (
                    system_prompt + "\n\nYour first response was not valid JSON. Return one compact JSON object now."
                )
                raw_response = judge.complete(
                    message_content,
                    system_prompt=retry_prompt,
                )
                try:
                    parsed = _normalize_judge_output(_json_from_text(raw_response))
                except (ValueError, json.JSONDecodeError) as parse_error:
                    elapsed = round(time.monotonic() - ep_started, 2)
                    entry.update(
                        {
                            "verdict": "uncertain",
                            "confidence": 0.0,
                            "violation_detected": False,
                            "criteria": [],
                            "evidence": [],
                            "temporal_events": [],
                            "camera_assessment": {},
                            "judge_status": "completed",
                            "parse_error": str(parse_error),
                            "raw_response": raw_response,
                            "sample_times_s": timestamps,
                            "elapsed_s": elapsed,
                        }
                    )
                    _emit(
                        f"[rubric] ep={media.episode_id} done verdict=uncertain (json_parse_error) "
                        f"sim={entry['sim_success']} elapsed={format_duration(elapsed)}",
                        level=logging.WARNING,
                    )
                    _record_and_flush(media.episode_id, entry)
                    return entry
            elapsed = round(time.monotonic() - ep_started, 2)
            entry.update(parsed)
            entry.update(
                {
                    "judge_status": "completed",
                    "raw_response": raw_response,
                    "sample_times_s": timestamps,
                    "elapsed_s": elapsed,
                }
            )
            _emit(
                f"[rubric] ep={media.episode_id} done verdict={entry['verdict']} "
                f"conf={entry['confidence']:.2f} violation={entry['violation_detected']} "
                f"sim={entry['sim_success']} elapsed={format_duration(elapsed)}"
            )
        except (MediaError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            elapsed = round(time.monotonic() - ep_started, 2)
            entry.update({"judge_status": "error", "error": str(exc), "elapsed_s": elapsed})
            _emit(
                f"[rubric] ep={media.episode_id} error ({format_duration(elapsed)}): {exc}",
                level=logging.ERROR,
            )
        _record_and_flush(media.episode_id, entry)
        return entry

    run_started = time.monotonic()
    worker_count = min(int(workers), max(1, len(episodes)))
    if worker_count == 1:
        entries = [_judge_episode(media) for media in episodes]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            entries = list(executor.map(_judge_episode, episodes))
    output["episodes"] = {media.episode_id: entry for media, entry in zip(episodes, entries)}
    output["elapsed_s"] = round(time.monotonic() - run_started, 2)

    output["report_dir"] = str(report_dir)
    atomic_write_json(report_path, output)

    error_count = sum(
        entry.get("judge_status", entry.get("review_status")) not in {"completed", "skipped"}
        for entry in output["episodes"].values()
    )
    result_file = path / "_result.json"
    if update_result and not rubric_configured:
        output["result_update"] = {"status": "skipped", "reason": "no task-specific rubric configured"}
    elif update_result and skip_update_on_error and error_count > 0:
        output["result_update"] = {
            "status": "skipped",
            "reason": "judge_errors_present",
            "error_count": error_count,
            "path": str(result_output_path or result_file),
        }
        _emit(
            f"[rubric] skipped updating _result.json because {error_count} episode(s) had judge errors",
            level=logging.WARNING,
        )
    elif update_result and result_file.exists() and not partial_eval:
        try:
            graded = apply_rubric_verdict(
                path,
                report_path=report_path,
                output_path=result_output_path,
                require_three_view=True,
                write=True,
                create_backup=create_backup,
            )
            meta = graded.get("rubric_eval", {})
            changes = meta.get("changes", {})
            target_result_path = str(
                Path(result_output_path).expanduser().resolve() if result_output_path else result_file
            )
            output["result_update"] = {
                "status": "updated",
                "path": target_result_path,
                "sim_backup_path": meta.get("sim_backup_path"),
                "sim_success_rate": graded.get("sim_success_rate"),
                "success_rate": graded.get("success_rate", 0.0),
                "sim_score": graded.get("sim_score"),
                "score": graded.get("score", 0.0),
                "overturned_count": changes.get("overturned_count", 0),
                "overturned_episodes": changes.get("overturned_episodes", []),
                "final_status_counts": meta.get("final_status_counts", {}),
            }
            overturned_ids = [item["episode_id"] for item in changes.get("overturned_episodes", [])]
            _emit(
                f"[rubric] result updated: {target_result_path} | "
                f"sr={graded.get('sim_success_rate')}->{graded.get('success_rate', 0.0):.4f} "
                f"score={graded.get('sim_score')}->{graded.get('score', 0.0):.2f} "
                f"overturned={len(overturned_ids)}"
                + (f" ({','.join(overturned_ids)})" if overturned_ids else "")
            )
        except ResultStoreError as exc:
            output["result_update"] = {"status": "error", "error": str(exc), "path": str(result_file)}
            _emit(f"[rubric] result update failed: {exc}", level=logging.ERROR)
    elif update_result and partial_eval:
        output["result_update"] = {"status": "skipped", "reason": "partial episode evaluation"}
    elif update_result:
        output["result_update"] = {"status": "skipped", "reason": "_result.json not found"}
    else:
        output["result_update"] = {"status": "disabled"}

    atomic_write_json(report_path, output)
    _write_markdown(report_dir / RUBRIC_REPORT_MD, output)
    return output


def _write_markdown(path: Path, output: dict[str, Any]) -> None:
    completed = [
        entry
        for entry in output["episodes"].values()
        if entry.get("judge_status", entry.get("review_status")) == "completed"
    ]
    lines = [
        "# RoboDojo VLM Rubric Evaluation",
        "",
        f"- Task: `{output['task']}`",
        f"- Judge model: `{output['judge_model']}`",
        f"- Sampling: every `{output['sampling']['frame_stride']}` frames "
        f"({output['sampling']['sample_fraction']:.3f} of source frames)",
        f"- Completed episodes: {len(completed)} / {len(output['episodes'])}",
    ]
    if "elapsed_s" in output:
        lines.append(f"- Elapsed: `{format_duration(output['elapsed_s'])}`")
    result_update = output.get("result_update")
    if isinstance(result_update, dict):
        status = result_update.get("status")
        if status == "updated":
            sim_sr = result_update.get("sim_success_rate")
            final_sr = result_update.get("success_rate")
            sim_sr_text = f"{sim_sr:.4f}" if isinstance(sim_sr, int | float) else "n/a"
            final_sr_text = f"{final_sr:.4f}" if isinstance(final_sr, int | float) else "n/a"
            overturned = [item["episode_id"] for item in result_update.get("overturned_episodes", [])]
            lines.append(f"- Result update: `updated` (`success_rate`: `{sim_sr_text}` -> `{final_sr_text}`)")
            if result_update.get("sim_backup_path"):
                lines.append(f"- Simulator backup: `{result_update['sim_backup_path']}`")
            if overturned:
                lines.append(f"- Overturned episodes ({len(overturned)}): `{', '.join(overturned)}`")
        elif status:
            reason = result_update.get("reason") or result_update.get("error") or ""
            lines.append(f"- Result update: `{status}`" + (f" ({reason})" if reason else ""))
    lines.extend(
        [
            "",
            "| Episode | Layout | Simulator | Verdict | Confidence | Violation | Judge Status |",
            "|---:|---:|---:|---|---:|---|---|",
        ]
    )
    for episode_id, entry in output["episodes"].items():
        verdict = entry.get("verdict", entry.get("vlm_verdict", ""))
        violation = entry.get("violation_detected", entry.get("shortcut_detected", ""))
        judge_status = entry.get("judge_status", entry.get("review_status", ""))
        lines.append(
            f"| {episode_id} | {entry.get('layout_id', '')} | {entry.get('sim_success', '')} | "
            f"{verdict} | {entry.get('confidence', '')} | {violation} | {judge_status} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "AVAILABLE_VIEW_PROMPT",
    "SYSTEM_PROMPT",
    "evaluate_run",
    "extract_criterion_ids",
]

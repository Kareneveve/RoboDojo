"""RoboDojo episode media discovery, synchronized frame sampling, and multi-view stitching."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

CAMERAS = ("head", "left_wrist", "right_wrist")
DEFAULT_PANEL_SIZE = (416, 312)
DEFAULT_HEADER_HEIGHT = 24
DEFAULT_JPEG_QUALITY = 82
VIDEO_RE = re.compile(
    r"^episode_(?P<episode>\d+)_cam_(?P<camera>head|left_wrist|right_wrist)(?:_[^.]*)?(?P<suffix>\.[^.]+)$"
)


class MediaError(RuntimeError):
    """Raised when an episode video or result media cannot be inspected."""


@dataclass(frozen=True)
class EpisodeMedia:
    episode_id: str
    cameras: dict[str, Path]
    result: dict[str, Any]


def load_result(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "_result.json"
    if not result_path.exists():
        # AGP showcase exports contain only episode videos. Reconstruct the
        # simulator summary from the filename status without fabricating scores.
        details: dict[str, dict[str, Any]] = {}
        for path in run_dir.iterdir():
            match = VIDEO_RE.match(path.name)
            if not match or path.suffix.lower() not in {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}:
                continue
            episode_id = str(int(match.group("episode")))
            status = path.stem.rsplit("_", 1)[-1].lower()
            detail = details.setdefault(episode_id, {"source": "video_filename", "camera_status": {}})
            detail["camera_status"][match.group("camera")] = status
        if details:
            for detail in details.values():
                statuses = list(detail["camera_status"].values())
                detail["success"] = bool(statuses) and all(status == "success" for status in statuses)
                detail["score"] = 1.0 if detail["success"] else 0.0
            return {"details": details, "source": "video_filename"}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MediaError(f"Cannot read {result_path}: {exc}") from exc
    if not isinstance(payload.get("details"), dict):
        raise MediaError(f"{result_path} has no details object")
    return payload


def discover_episodes(run_dir: Path, result_payload: dict[str, Any]) -> list[EpisodeMedia]:
    """Bind multi-camera videos to sequential result keys, preserving layout metadata."""

    videos: dict[str, dict[str, Path]] = {}
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        match = VIDEO_RE.match(path.name)
        if not match or path.suffix.lower() not in {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}:
            continue
        episode = str(int(match.group("episode")))
        videos.setdefault(episode, {}).setdefault(match.group("camera"), path)

    episodes: list[EpisodeMedia] = []
    detail_map = result_payload["details"]
    normalized_details = {str(int(key)): value for key, value in detail_map.items() if str(key).lstrip("-").isdigit()}
    episode_ids = sorted(set(videos) | set(normalized_details), key=int)
    for episode_id in episode_ids:
        result = normalized_details.get(episode_id, {})
        episodes.append(
            EpisodeMedia(
                episode_id=episode_id,
                cameras=videos.get(episode_id, {}),
                result=dict(result) if isinstance(result, dict) else {},
            )
        )
    return episodes


def _run(command: list[str]) -> bytes:
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise MediaError(f"Command failed: {' '.join(command)}\n{detail}") from exc
    return completed.stdout


def video_duration(path: Path) -> float:
    output = (
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ]
        )
        .decode("utf-8", errors="replace")
        .strip()
    )
    try:
        duration = float(output)
    except ValueError as exc:
        raise MediaError(f"Invalid duration from ffprobe for {path}: {output!r}") from exc
    if duration <= 0:
        raise MediaError(f"Video has non-positive duration: {path}")
    return duration


def _parse_frame_rate(value: object) -> float:
    text = str(value or "").strip()
    if not text or text in {"N/A", "0/0"}:
        return 0.0
    try:
        rate = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return rate if rate > 0 else 0.0


def video_frame_info(path: Path) -> tuple[int, float]:
    """Return decoded frame count and average frame rate for one video."""

    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,avg_frame_rate,r_frame_rate,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(output.decode("utf-8", errors="replace"))
        stream = payload["streams"][0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise MediaError(f"Cannot read frame metadata from {path}") from exc

    frame_rate = _parse_frame_rate(stream.get("avg_frame_rate"))
    if not frame_rate:
        frame_rate = _parse_frame_rate(stream.get("r_frame_rate"))
    if not frame_rate:
        raise MediaError(f"Video has no usable frame rate: {path}")

    frame_count = 0
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        try:
            frame_count = int(value)
        except (TypeError, ValueError):
            continue
        if frame_count > 0:
            break
    if frame_count <= 0:
        try:
            duration = float(stream.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        frame_count = int(round(duration * frame_rate))
    if frame_count <= 0:
        raise MediaError(f"Video has no readable frames: {path}")
    return frame_count, frame_rate


def extract_frame(path: Path, timestamp_s: float) -> bytes:
    data = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp_s):.6f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "4",
            "pipe:1",
        ]
    )
    if data or timestamp_s <= 0.0:
        return data
    # Guard against tail-frame timestamp rounding past EOF on variable-rate streams.
    return _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-sseof",
            "-0.1",
            "-i",
            str(path),
            "-update",
            "1",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "4",
            "pipe:1",
        ]
    )


def synchronized_frames(
    media: EpisodeMedia,
    sample_fraction: float = 0.10,
    cameras: tuple[str, ...] = CAMERAS,
    *,
    metadata: dict[str, Any] | None = None,
    extract_workers: int = 8,
) -> tuple[list[float], dict[str, list[bytes]]]:
    """Uniformly downsample synchronized videos at a fixed frame stride.

    A ten percent sample uses stride ten, which means frame indices
    ``0, 10, 20, ...`` plus the final common frame (when not already included)
    so short tail-end actions or violations are never clipped.
    """

    missing = sorted(set(cameras) - set(media.cameras))
    if missing:
        raise MediaError(f"episode {media.episode_id} missing cameras: {', '.join(missing)}")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be greater than 0 and at most 1")

    frame_stride = max(1, math.ceil(1 / sample_fraction))
    stream_info = {camera: video_frame_info(media.cameras[camera]) for camera in cameras}
    common_frame_count = min(frame_count for frame_count, _ in stream_info.values())
    frame_indices = list(range(0, common_frame_count, frame_stride))
    last_frame = common_frame_count - 1
    if last_frame > 0 and frame_indices[-1] != last_frame:
        frame_indices.append(last_frame)
    common_rate = min(rate for _, rate in stream_info.values())
    timestamps = [index / common_rate for index in frame_indices]
    if metadata is not None:
        metadata.update(
            {
                "sample_fraction": sample_fraction,
                "frame_stride": frame_stride,
                "sample_indices": frame_indices,
                "source_frame_counts": {camera: frame_count for camera, (frame_count, _) in stream_info.items()},
                "source_frame_rates": {camera: rate for camera, (_, rate) in stream_info.items()},
            }
        )
    jobs = [
        (camera, pos, media.cameras[camera], index / stream_info[camera][1])
        for camera in cameras
        for pos, index in enumerate(frame_indices)
    ]
    frames: dict[str, list[bytes]] = {camera: [b""] * len(frame_indices) for camera in cameras}
    worker_count = max(1, min(int(extract_workers), len(jobs)))
    if worker_count == 1:
        for camera, pos, path, ts in jobs:
            frames[camera][pos] = extract_frame(path, ts)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            extracted = list(executor.map(lambda job: extract_frame(job[2], job[3]), jobs))
        for (camera, pos, _, _), frame_bytes in zip(jobs, extracted):
            frames[camera][pos] = frame_bytes
    return timestamps, frames


def stitch_camera_frames(
    camera_frames: Mapping[str, bytes],
    cameras: tuple[str, ...] = CAMERAS,
    *,
    panel_size: tuple[int, int] = DEFAULT_PANEL_SIZE,
    header_height: int = DEFAULT_HEADER_HEIGHT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Stitch synchronized camera views horizontally into one labeled composite JPEG."""

    if not cameras:
        raise ValueError("cameras must not be empty")
    missing = [camera for camera in cameras if camera not in camera_frames]
    if missing:
        raise MediaError(f"Missing frame bytes for cameras: {', '.join(missing)}")
    if len(cameras) == 1:
        return camera_frames[cameras[0]]

    panel_w, panel_h = panel_size
    if panel_w < 1 or panel_h < 1 or header_height < 0:
        raise ValueError("panel_size must be positive and header_height non-negative")

    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise MediaError("Pillow is required to stitch multi-view camera frames") from exc

    canvas = Image.new("RGB", (panel_w * len(cameras), panel_h + header_height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, camera in enumerate(cameras):
        x0 = index * panel_w
        try:
            with Image.open(io.BytesIO(camera_frames[camera])) as raw_image:
                panel = ImageOps.pad(
                    raw_image.convert("RGB"),
                    (panel_w, panel_h),
                    method=Image.Resampling.LANCZOS,
                    color=(0, 0, 0),
                )
        except Exception as exc:
            raise MediaError(f"Cannot decode frame bytes for camera {camera}: {exc}") from exc
        canvas.paste(panel, (x0, header_height))
        if header_height > 0:
            draw.rectangle((x0, 0, x0 + panel_w - 1, header_height - 1), fill=(20, 20, 20))
            draw.text((x0 + 6, max(0, (header_height - 11) // 2)), camera, fill=(255, 255, 0))
        if index > 0:
            draw.line((x0, 0, x0, panel_h + header_height - 1), fill=(200, 200, 200), width=2)

    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def image_data_url(frame: bytes, *, mime_type: str = "image/jpeg") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(frame).decode('ascii')}"


__all__ = [
    "CAMERAS",
    "DEFAULT_HEADER_HEIGHT",
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_PANEL_SIZE",
    "EpisodeMedia",
    "MediaError",
    "VIDEO_RE",
    "discover_episodes",
    "extract_frame",
    "image_data_url",
    "load_result",
    "stitch_camera_frames",
    "synchronized_frames",
    "video_duration",
    "video_frame_info",
]

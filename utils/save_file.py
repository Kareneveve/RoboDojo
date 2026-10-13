import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
from typing import Any

import numpy as np

_WRITE_POLL_S = 0.1


def _start_ffmpeg(command: list[str]) -> subprocess.Popen:
    """Keep system ffmpeg isolated from Conda/Isaac library overrides."""
    executable = shutil.which(command[0])
    env = os.environ.copy()
    if executable is not None:
        command = [executable, *command[1:]]
        if Path(executable).resolve().parent in (Path("/usr/bin"), Path("/bin")):
            env.pop("LD_LIBRARY_PATH", None)
    return subprocess.Popen(command, stdin=subprocess.PIPE, env=env)


def format_video_saved_message(
    path: str,
    n_frames: int,
    width: int,
    height: int,
    fps: float,
) -> str:
    return (
        f"🎬 Video is saved to `{path}`, containing "
        f"\033[94m{n_frames}\033[0m frames at {width}×{height} "
        f"resolution and {fps} FPS."
    )


class VideoStreamWriter:
    """Stream owned frame snapshots to ffmpeg on a background thread.

    One producer calls ``append`` and ``request_close`` / ``close``. Reusable
    host buffers bound memory; a full pool waits instead of dropping frames.
    ``abort`` kills the encoder before joining a possibly blocked pipe writer.
    """

    _STOP = object()

    def __init__(
        self,
        out_path: str,
        height: int,
        width: int,
        channels: int,
        fps: float = 30.0,
        is_rgb: bool = True,
        queue_size: int = 8,
        encoder_preset: str | None = "fast",
    ) -> None:
        if int(queue_size) != queue_size or queue_size <= 0:
            raise ValueError("queue_size must be a positive integer")
        if channels == 3:
            pixel_format = "rgb24" if is_rgb else "bgr24"
        elif channels == 4:
            pixel_format = "rgba"
        else:
            raise ValueError(f"Unsupported channel count for video: {channels}")
        self.out_path = out_path
        self.height = height
        self.width = width
        self.channels = channels
        self.fps = fps
        self.queue_size = int(queue_size)
        self.n_frames = 0
        self.written_frames = 0
        self._closed = False
        self._close_requested = False
        self._aborting = threading.Event()
        self._worker_error: BaseException | None = None
        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=self.queue_size)
        self._free_buffers: queue.LifoQueue[np.ndarray] = queue.LifoQueue()
        for _ in range(self.queue_size):
            self._free_buffers.put(np.empty((height, width, channels), dtype=np.uint8))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
        ]
        if encoder_preset:
            command.extend(["-preset", encoder_preset])
        command.append(out_path)
        self.proc = _start_ffmpeg(command)
        self._worker = threading.Thread(
            target=self._write_loop,
            name=f"video-writer:{os.path.basename(out_path)}",
            daemon=True,
        )
        self._worker.start()

    def _raise_if_worker_failed(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(f"Background video write failed for `{self.out_path}`") from self._worker_error
        if not self._worker.is_alive():
            raise RuntimeError(f"Background video writer stopped for `{self.out_path}`")

    def _acquire_buffer(self) -> np.ndarray:
        while True:
            if self._closed:
                raise RuntimeError("Cannot append to a closed VideoStreamWriter.")
            self._raise_if_worker_failed()
            try:
                return self._free_buffers.get(timeout=_WRITE_POLL_S)
            except queue.Empty:
                pass

    def _write_loop(self) -> None:
        # Only this thread accesses stdin, including its buffered close/flush.
        try:
            while True:
                payload = self._queue.get()
                try:
                    if payload is self._STOP:
                        return
                    if not self._aborting.is_set():
                        self.proc.stdin.write(memoryview(payload).cast("B"))
                        self.written_frames += 1
                finally:
                    if isinstance(payload, np.ndarray):
                        self._free_buffers.put(payload)
                    self._queue.task_done()
        except BaseException as exc:
            self._worker_error = exc
        finally:
            try:
                self.proc.stdin.close()
            except BaseException as exc:
                if self._worker_error is None:
                    self._worker_error = exc

    def append(self, frame: np.ndarray) -> None:
        if self.proc is None or self._closed:
            raise RuntimeError("Cannot append to a closed VideoStreamWriter.")
        if frame.shape != (self.height, self.width, self.channels):
            raise ValueError(
                f"Frame shape {tuple(frame.shape)} does not match writer ({self.height}x{self.width}x{self.channels})."
            )
        # Isaac may reuse frame immediately after append; the worker needs its
        # own snapshot. Pool capacity also bounds the frame currently encoding.
        buffer = self._acquire_buffer()
        try:
            np.copyto(buffer, frame, casting="unsafe")
            self._raise_if_worker_failed()
            if self._closed:
                raise RuntimeError("Cannot append to a closed VideoStreamWriter.")
            self._queue.put_nowait(buffer)
        except BaseException:
            self._free_buffers.put(buffer)
            raise
        self.n_frames += 1

    def request_close(self) -> None:
        """Queue EOF; call for every camera before waiting in ``close``."""
        if self.proc is None or self._close_requested:
            return
        self._closed = True
        while True:
            self._raise_if_worker_failed()
            try:
                self._queue.put(self._STOP, timeout=_WRITE_POLL_S)
                self._close_requested = True
                return
            except queue.Full:
                pass

    def close(self, *, announce: bool = True) -> None:
        if self.proc is None:
            return
        try:
            self.request_close()
            self._worker.join()
            if self._worker_error is not None:
                raise OSError(f"ffmpeg write failed for `{self.out_path}`.") from self._worker_error
            if self.proc.wait() != 0:
                raise OSError(f"ffmpeg failed while finalizing `{self.out_path}`.")
            if self.written_frames != self.n_frames:
                raise OSError(
                    f"Video frame mismatch for `{self.out_path}`: "
                    f"queued={self.n_frames}, written={self.written_frames}"
                )
        except BaseException as exc:
            self.abort()
            if isinstance(exc, Exception) and self._worker_error is not None:
                raise OSError(f"ffmpeg write failed for `{self.out_path}`.") from self._worker_error
            raise
        finally:
            self.proc = None
        if announce:
            print(
                format_video_saved_message(
                    self.out_path,
                    self.n_frames,
                    self.width,
                    self.height,
                    self.fps,
                )
            )

    def abort(self) -> None:
        """Kill the ffmpeg process and remove the partial output file."""
        self._aborting.set()
        self._closed = True
        if self.proc is not None:
            # Killing first unblocks a worker waiting in stdin.write/close.
            try:
                self.proc.kill()
            except Exception:
                pass
            try:
                self.request_close()
            except Exception:
                pass
            self._worker.join()
            try:
                self.proc.wait()
            except Exception:
                pass
            self.proc = None
        try:
            if os.path.exists(self.out_path):
                os.remove(self.out_path)
        except Exception:
            pass


def save_json(
    data: Any,
    path: str | os.PathLike,
    overwrite: bool = True,
    make_dirs: bool = True,
    sort_keys: bool = False,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    p = Path(path)

    if make_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists and overwrite=False")

    tmp = p.with_suffix(p.suffix + ".tmp")

    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(
                data,
                f,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                indent=indent,
            )
            f.write("\n")
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            raise

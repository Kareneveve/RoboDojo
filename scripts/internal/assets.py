#!/usr/bin/env python3
"""Download selected benchmark assets from a shared Git/LFS repository."""

from __future__ import annotations

import argparse
import errno
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("RoboDojo", "MiVerse")
SHARED = {"Material", "Background", "Sensor", "Traj"}
REQUIRED = {
    "RoboDojo": (
        "Object/RoboDojo", "Eval_Layout/RoboDojo/arx_x5", "Robots/x5",
        "Robots/franka", "Room/Simple_Room", "Room/Simple_Room_nolight",
    ),
    "MiVerse": (
        "Object/MiVerse", "Eval_Layout/MiVerse/miverse", "Robots/m92uw", "Room/MiVerse",
    ),
}
ROBOT_FILES = {
    "RoboDojo": ("Robots/x5/ARX.usd", "Robots/x5/robot_config.yml",
                 "Robots/franka/Franka.usd", "Robots/franka/robot_config.yml"),
    "MiVerse": ("Robots/m92uw/M92UW.usd", "Robots/m92uw/robot_config.yml"),
}
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def git(cache, *args, capture=False, input=None):
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.run(
        ["git", "-C", str(cache), *args], env=env, input=input, check=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def selected(path, benches):
    parts = Path(path).parts
    if parts[0] in SHARED:
        return True
    if len(parts) < 3:
        return False
    if parts[0] in {"Object", "Eval_Layout"}:
        return parts[1] in benches
    if parts[0] == "Robots":
        return ("MiVerse" in benches and parts[1] == "m92uw") or (
            "RoboDojo" in benches and parts[1] in {"x5", "franka"}
        )
    if parts[0] == "Room":
        return ("MiVerse" if parts[1] == "MiVerse" else "RoboDojo") in benches
    return False


def read_manifest(cache, revision, benches):
    """Read Git blob sizes and declared LFS sizes without downloading payloads."""
    tree = git(cache, "ls-tree", "-r", "-l", "-z", revision, "--", "Assets", capture=True)
    manifest = {}
    small_blobs = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        mode, kind, oid, size = meta.split()
        path = raw_path.decode("utf-8").removeprefix("Assets/")
        if not selected(path, benches):
            continue
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise ValueError(f"Unsupported asset entry (expected regular file): Assets/{path}")
        manifest[path] = {"size": int(size), "lfs": False}
        if int(size) < 1024:
            small_blobs.setdefault(oid.decode(), []).append(path)
    if small_blobs:
        output = git(cache, "cat-file", "--batch", capture=True,
                     input=("\n".join(small_blobs) + "\n").encode())
        stream = io.BytesIO(output)
        for paths in small_blobs.values():
            _, _, size = stream.readline().split()
            data = stream.read(int(size))
            stream.read(1)
            if data.startswith(LFS_HEADER):
                match = re.search(rb"^size (\d+)$", data, re.MULTILINE)
                oid = re.search(rb"^oid sha256:[0-9a-f]{64}$", data, re.MULTILINE)
                if not match or not oid:
                    raise ValueError(f"Invalid LFS pointer: Assets/{paths[0]}")
                for path in paths:
                    manifest[path] = {"size": int(match[1]), "lfs": True}
    return manifest


def validate_source(manifest, benches):
    required = ["Material", "Background"]
    for bench in benches:
        required.extend(REQUIRED[bench])
    missing = []
    for prefix in required:
        if not any(
            path.startswith(prefix + "/") and entry["size"] > 0
            and Path(path).name not in {".gitkeep", ".DS_Store"}
            and Path(path).suffix.lower() not in {".md", ".txt"}
            for path, entry in manifest.items()
        ):
            missing.append("Assets/" + prefix)
    for bench in benches:
        missing.extend("Assets/" + path for path in ROBOT_FILES[bench]
                       if path not in manifest or manifest[path]["size"] == 0)
    if missing:
        raise ValueError(
            "Selected source/revision is missing required benchmark assets:\n  "
            + "\n  ".join(sorted(set(missing)))
            + "\nChoose a source/revision containing the selected benchmarks "
            "(HF_REPO_URL / HF_REPO_ID / HF_REVISION), or select --bench RoboDojo or MiVerse."
        )


def valid_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as handle:
        return not handle.read(len(LFS_HEADER)).startswith(LFS_HEADER)


def placeholder_file(path):
    if not path.is_file() or path.is_symlink():
        return False
    with path.open("rb") as handle:
        return path.stat().st_size == 0 or handle.read(len(LFS_HEADER)) == LFS_HEADER


def validate_files(root, manifest):
    invalid = [str(root / path) for path, entry in manifest.items() if not valid_file(root / path, entry)]
    if invalid:
        raise ValueError("Missing, truncated, or unresolved LFS assets:\n  " + "\n  ".join(invalid[:10]))
    for path in manifest:
        if path.startswith("Eval_Layout/") and path.endswith(".json"):
            try:
                data = json.loads((root / path).read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, OSError) as exc:
                raise ValueError(f"Invalid layout Assets/{path}: {exc}") from exc


def file_identity(path):
    stat = path.lstat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def publish(cache_assets, target, manifest, installed):
    """Fill missing assets without replacing local content or other benchmarks."""
    if target.is_symlink() and not target.exists() and target.resolve() != cache_assets.resolve():
        raise ValueError(f"Assets is a broken external symlink: {target}; restore its target first.")
    if target.resolve() == cache_assets.resolve():
        return {}  # Compatibility with the previous whole-cache Assets symlink.
    target.mkdir(parents=True, exist_ok=True)
    pending = []
    for name, entry in manifest.items():
        source = cache_assets / name
        dest = target / name
        if dest.exists() or dest.is_symlink():
            managed = installed.get(name) == file_identity(dest)
            if dest.is_symlink() and dest.resolve() == source.resolve():
                managed = True
            if managed or placeholder_file(dest):
                pass
            elif valid_file(dest, entry):
                continue
            else:
                raise ValueError(
                    f"Existing local asset differs from the source: {dest}; preserve/move it before retrying."
                )
        pending.append(name)
    # Check conflicts before changing any published files. Keep real files in
    # Assets so an Assets-only Docker bind mount needs no external cache paths.
    published = {}
    for name in pending:
        source = cache_assets / name
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.link(source, dest)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source, dest)
        published[name] = file_identity(dest)
    return published


def download(root, requested):
    source = os.environ.get("HF_REPO_URL") or (
        "https://huggingface.co/datasets/" + os.environ.get("HF_REPO_ID", "RoboDojo-Benchmark/RoboDojo")
    )
    revision = os.environ.get("HF_REVISION", "main")
    cache = root / ".cache/robodojo_assets_repo"
    target = root / "Assets"
    subprocess.run(["git", "lfs", "version"], check=True, stdout=subprocess.DEVNULL)
    if not (cache / ".git").is_dir():
        if cache.exists() and any(cache.iterdir()):
            raise ValueError(
                f"Asset cache exists but is not a Git repository: {cache}; preserve/move it before retrying."
            )
        cache.mkdir(parents=True, exist_ok=True)
        git(cache, "init", "--quiet")
        git(cache, "remote", "add", "origin", source)
    elif git(cache, "remote", "get-url", "origin", capture=True).decode().strip() != source:
        raise ValueError(
            f"Asset cache belongs to a different source: {cache}; preserve/move it before changing HF_REPO_URL."
        )
    state_path = cache / ".git/robodojo_assets.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    previous = state.get("benchmarks", [])
    # Preserve assets installed by the old script, which has no state file.
    if not state_path.exists() and target.is_symlink() and target.resolve() == (cache / "Assets").resolve():
        for bench in BENCHMARKS:
            package = cache / "Assets/Object" / bench
            if package.is_dir() and any(
                path.is_file() and path.stat().st_size > 0 and not placeholder_file(path)
                and path.suffix.lower() not in {".md", ".txt"} and path.name != ".gitkeep"
                for path in package.rglob("*")
            ):
                previous.append(bench)
    benches = [bench for bench in BENCHMARKS if bench in set(requested) | set(previous)]
    print(f"[assets] requested={','.join(requested)}; retaining={','.join(benches)}; revision={revision}", flush=True)
    git(cache, "fetch", "--depth", "1", "origin", revision)
    commit = git(cache, "rev-parse", "FETCH_HEAD", capture=True).decode().strip()
    manifest = read_manifest(cache, commit, benches)
    validate_source(manifest, benches)
    # Cone mode includes parent files; explicit LFS filters below limit payloads.
    directories = sorted({"Assets/" + "/".join(Path(path).parts[:2])
                          if Path(path).parts[0] not in SHARED else "Assets/" + Path(path).parts[0]
                          for path in manifest})
    git(cache, "sparse-checkout", "set", "--cone", *directories)
    git(cache, "checkout", "--detach", commit)
    # Restore deleted files and pointer placeholders before the filtered LFS pull.
    broken = ["Assets/" + path for path, entry in manifest.items()
              if not valid_file(cache / "Assets" / path, entry)]
    if broken:
        for name in broken:
            path = cache / name
            if (path.exists() or path.is_symlink()) and not placeholder_file(path):
                raise ValueError(
                    f"Existing cached asset differs from the source: {path}; preserve/move it before retrying."
                )
        git(cache, "restore", "--source", commit, "--worktree", "--pathspec-from-file=-",
            "--pathspec-file-nul", input=("\0".join(broken) + "\0").encode())
    git(cache, "lfs", "install", "--local")
    git(cache, "lfs", "pull", "--include=" + ",".join(path + "/**" for path in directories), "--exclude=")
    validate_files(cache / "Assets", manifest)
    published = publish(cache / "Assets", target, manifest, state.get("files", {}))
    validate_files(target, manifest)
    state_path.write_text(json.dumps({"benchmarks": benches, "commit": commit, "files": published}, indent=2) + "\n")
    print(f"[assets] Assets directory is ready for {', '.join(requested)}: {target}", flush=True)
    print("[assets] Run evaluate.sh doctor for the task, seed and episode count before evaluation.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog="bash scripts/init_assets.sh", description=__doc__,
        epilog="Source: HF_REPO_URL or HF_REPO_ID (default RoboDojo-Benchmark/RoboDojo); HF_REVISION defaults to main.",
    )
    parser.add_argument("--bench", choices=("all", *BENCHMARKS), default="all",
                        help="Assets to download (default: all). Previously downloaded benchmarks are retained.")
    args = parser.parse_args()
    try:
        download(ROOT, list(BENCHMARKS) if args.bench == "all" else [args.bench])
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[assets] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

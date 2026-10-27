#!/usr/bin/env python3
"""Download the MiVerse ModelScope dataset with an explicitly supplied token."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatchcase
import getpass
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path, PurePosixPath
import re
import sys

DEFAULT_REPO = "ridiculouslzx/MiVerse"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "MiVerse"
INSTALL_HINT = "python3 -m pip install -U 'modelscope-hub>=0.5.0'"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def file_workers(value: str) -> int:
    number = positive_int(value)
    if number > 16:
        raise argparse.ArgumentTypeError("must be between 1 and 16 (SDK stream limit)")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download MiVerse from ModelScope, preserving its directory structure.",
        epilog=("Token priority: --token / --token-file, MODELSCOPE_API_TOKEN, then a "
                "hidden terminal prompt. Install the optional SDK with: " + INSTALL_HINT),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help=f"dataset ID (default: {DEFAULT_REPO})")
    parser.add_argument("--revision", default="master", help="branch/tag/commit (default: master)")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_DIR,
                        help=f"download destination (default: {DEFAULT_DIR})")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--token", help="ModelScope access token; prefer the prompt or environment")
    auth.add_argument("--token-file", type=Path, help="read the token from a UTF-8 text file")
    parser.add_argument("--include", action="append", default=[], metavar="GLOB",
                        help="include matching remote paths; repeatable, trailing / selects a directory")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                        help="exclude matching remote paths; repeatable, takes priority over --include")
    parser.add_argument("--max-workers", type=positive_int, default=8,
                        help="parallel file downloads (default: 8)")
    parser.add_argument("--file-workers", type=file_workers, default=8,
                        help="range download threads per large file, 1-16 (default: 8; 1 disables)")
    parser.add_argument("--parallel-threshold-mb", type=positive_int, default=64,
                        help="use range threads for files larger than this many MiB (default: 64)")
    parser.add_argument("--force", action="store_true", help="redownload selected files even if cached")
    parser.add_argument("--list", action="store_true", help="list selected files and sizes without downloading")
    return parser.parse_args(argv)


def read_token(args) -> str:
    if args.token is not None:
        token = args.token
    elif args.token_file is not None:
        token = args.token_file.expanduser().read_text(encoding="utf-8")
    else:
        token = os.environ.get("MODELSCOPE_API_TOKEN", "")
        if not token.strip() and sys.stdin.isatty():
            token = getpass.getpass("ModelScope token: ")
    token = token.strip()
    if not token:
        raise ValueError("A token is required. Set MODELSCOPE_API_TOKEN or use --token-file.")
    if any(character.isspace() for character in token):
        raise ValueError("The token must be a single value without whitespace.")
    return token


def configure_downloads(args):
    # The SDK reads these settings once at import time. Configure before create_api.
    os.environ["MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS"] = str(args.file_workers)
    os.environ["MODELSCOPE_DOWNLOAD_PARALLEL_THRESHOLD_MB"] = str(args.parallel_threshold_mb)
    # Fixed boundaries allow partial ranges to be reused when thread counts change.
    os.environ["MODELSCOPE_DOWNLOAD_PART_SIZE_MB"] = "64"


def create_api(token):
    try:
        installed = version("modelscope-hub")
    except PackageNotFoundError:
        raise RuntimeError(f"Missing ModelScope SDK. Install it with: {INSTALL_HINT}") from None
    # Public download_file(file_size=...) needs 0.5.0 to enable per-file range threads.
    release = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[\w.]+)?", installed)
    if release is None or tuple(map(int, release.groups())) < (0, 5, 0):
        raise RuntimeError(f"modelscope-hub {installed} is unsupported; run: {INSTALL_HINT}")
    from modelscope_hub import HubApi

    # Supplying token here authenticates in memory; login() would persist credentials.
    return HubApi(token=token, endpoint="https://modelscope.cn")


def matches(path, patterns):
    return any(fnmatchcase(path, pattern + "*" if pattern.endswith("/") else pattern)
               for pattern in patterns)


def local_path(root, remote_path):
    path = PurePosixPath(remote_path)
    if not remote_path or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"Invalid remote file path: {remote_path!r}")
    destination = root / path
    if not destination.resolve().is_relative_to(root):
        raise ValueError(f"Download path escapes the destination: {remote_path!r}")
    return destination


def download_one(api, args, root, info):
    destination = local_path(root, info.path)
    file_size = info.size
    partial = destination.with_suffix(destination.suffix + ".incomplete")
    if not args.force and partial.is_file() and partial.stat().st_size > 0:
        # SDK range downloads cannot reuse a single-stream .incomplete prefix.
        # Finish these files in their original mode instead of discarding progress.
        file_size = None
        print(f"Resuming existing partial with one connection: {info.path}", flush=True)
    # Without a remote checksum, the SDK accepts any existing local file as cached,
    # even when it belongs to another revision. Refresh those files instead.
    api.download_file(
        args.repo_id, repo_type="dataset", file_path=info.path,
        revision=args.revision, local_dir=root, force=args.force or not info.sha256,
        expected_sha256=info.sha256 or None,
        file_size=file_size,
    )
    if not destination.is_file() or destination.stat().st_size != info.size:
        raise RuntimeError(f"Incomplete download: {info.path} (expected {info.size} bytes)")
    return info.path


def run(args, token):
    configure_downloads(args)
    api = create_api(token)
    root = args.local_dir.expanduser().resolve()
    files = api.list_repo_files(args.repo_id, repo_type="dataset", revision=args.revision, recursive=True)
    selected = sorted(
        (info for info in files if info.type != "tree"
         and (not args.include or matches(info.path, args.include))
         and not matches(info.path, args.exclude)),
        key=lambda info: info.path,
    )
    if not selected:
        raise RuntimeError("No files matched the selection; check the repository, revision and filters.")
    for info in selected:
        local_path(root, info.path)
    total_size = sum(info.size for info in selected)
    print(f"Dataset: {args.repo_id} @ {args.revision}")
    print(f"Selected: {len(selected)} files, {total_size:,} bytes")
    if args.list:
        for info in selected:
            print(f"{info.size:>14,}  {info.path}")
        return
    root.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {root}")
    print(f"Concurrency: {args.max_workers} files x up to {args.file_workers} range threads "
          f"(up to {args.max_workers * args.file_workers} download connections); "
          f"threshold > {args.parallel_threshold_mb} MiB, parts 64 MiB", flush=True)
    # download_repo() logs individual failures and returns success. Keep each future's
    # exception visible here, and send exact paths rather than the SDK's legacy regex globs.
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(download_one, api, args, root, info) for info in selected]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                print(f"[{count}/{len(selected)}] {future.result()}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    print(f"Download complete: {len(selected)} files in {root}")


def main(argv=None):
    args = parse_args(argv)
    token = ""
    try:
        token = read_token(args)
        run(args, token)
    except (KeyboardInterrupt, EOFError):
        print("Download interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc)
        if token:
            message = message.replace(token, "[REDACTED]")
        print(f"Error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

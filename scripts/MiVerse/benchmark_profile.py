#!/usr/bin/env python3
"""Resolve MiVerse profile and result-directory names without simulation imports."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_profile(dataset: str = "MiVerse", env_cfg: str = "") -> tuple[str, str, str]:
    if dataset not in ("", "MiVerse"):
        raise ValueError("This CLI only supports --dataset MiVerse.")
    env_cfg = env_cfg or "miverse"
    path = ROOT_DIR / "env_cfg" / f"{env_cfg}.yml"
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"Invalid eval profile: {path}")
    if profile.get("benchmark") != "MiVerse":
        raise ValueError(f"--env-cfg {env_cfg} must declare benchmark: MiVerse.")
    config_name = profile.get("config_name")
    if not isinstance(config_name, str) or not config_name.strip():
        raise ValueError(f"Missing config_name in {path}")
    return "MiVerse", env_cfg, config_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MiVerse",), default="MiVerse")
    parser.add_argument("--env-cfg", default="miverse", help="MiVerse env_cfg stem (default: miverse).")
    args = parser.parse_args()
    try:
        resolved = resolve_profile(args.dataset, args.env_cfg)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print("\t".join(resolved))


if __name__ == "__main__":
    main()

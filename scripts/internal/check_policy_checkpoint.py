#!/usr/bin/env python3
"""Check checkpoint paths using XPolicyLab's resolver without loading a model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def enabled(value):
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def policy_path(value, policy_dir):
    path = Path(value).expanduser()
    return path if path.is_absolute() else policy_dir / path


def check_checkpoint(policy_dir, *, bench, env_cfg, task, seed, action_type="ee", checkpoint=None):
    policy_dir = Path(policy_dir).resolve()
    config = yaml.safe_load((policy_dir / "deploy.yml").read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("deploy.yml must contain a mapping")
    policy = policy_dir.name  # The server launcher also overrides policy_name with this basename.
    config.update(bench_name=bench, env_cfg_type=env_cfg, task_name=task,
                  seed=seed, action_type=action_type, policy_name=policy)
    if checkpoint is not None:
        config["ckpt_name"] = checkpoint
    if policy == "demo_policy":
        return "WARN", "demo_policy is weight-free; no checkpoint is required"

    from XPolicyLab.utils.checkpoint_resolver import (
        DEFAULT_EXPLICIT_KEYS,
        candidate_checkpoint_roots,
        resolve_checkpoint_root,
    )

    explicit_keys = DEFAULT_EXPLICIT_KEYS
    checkpoints_dir = policy_dir / "checkpoints"
    if policy == "Pi_05":
        explicit_keys = ("model_path", "checkpoint_path")
    elif policy == "OpenWAM":
        explicit_keys = ("ckpt_dir", "checkpoint_path", "ckpt_path", "model_dir")
        for variable, key in (("OPENWAM_CKPT_DIR", "ckpt_dir"),
                              ("OPENWAM_ALLOW_DUMMY_POLICY", "allow_dummy_policy")):
            if os.environ.get(variable):
                config[key] = os.environ[variable]
        if enabled(config.get("allow_dummy_policy")):
            return "WARN", "OpenWAM dummy policy skips checkpoint loading (debug only)"
    elif policy == "MM_ABC" and enabled(config.get("mock")):
        return "WARN", "MM_ABC mock mode skips checkpoint loading (debug only)"
    elif policy == "FastWAM":
        # Its shell launcher overrides deploy checkpoint paths. Check the
        # selected file when explicit, otherwise the shared run-root only;
        # step selection and dataset-stat contents remain adapter checks.
        if os.environ.get("FASTWAM_ALLOW_DUMMY_POLICY") == "true":
            return "WARN", "FastWAM dummy policy skips checkpoint loading (debug only)"
        explicit = os.environ.get("FASTWAM_CHECKPOINT_PATH")
        if explicit:
            config["model_path"] = str(policy_path(explicit, policy_dir))
            config["ckpt_name"] = None
            explicit_keys = ("model_path",)
        else:
            checkpoints_dir = policy_path(os.environ.get("FASTWAM_CKPT_ROOT") or "checkpoints", policy_dir)
            explicit_keys = ()
            if os.environ.get("FASTWAM_CKPT_SETTING"):
                config["model_path"] = str(checkpoints_dir) + "/" + os.environ["FASTWAM_CKPT_SETTING"]
                config["ckpt_name"] = None
                explicit_keys = ("model_path",)

    options = {"policy_dir": policy_dir, "explicit_keys": explicit_keys}
    if not candidate_checkpoint_roots(config, checkpoints_dir, **options):
        return "WARN", "no checkpoint selected; pass --ckpt or configure a checkpoint path in deploy.yml"
    resolved = resolve_checkpoint_root(config, checkpoints_dir, **options)
    return "PASS", f"{resolved} (path exists; model weights and adapter artifacts are not checked)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--env-cfg", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-type", choices=("joint", "ee"), default="ee")
    parser.add_argument("--ckpt")
    args = parser.parse_args()
    try:
        status, message = check_checkpoint(
            args.policy_dir, bench=args.bench, env_cfg=args.env_cfg, task=args.task,
            seed=args.seed, action_type=args.action_type, checkpoint=args.ckpt,
        )
    except (OSError, ValueError, ImportError, yaml.YAMLError) as exc:
        print("; ".join(str(exc).splitlines()))
        return 1
    print(status + "\t" + message)
    return 0


if __name__ == "__main__":
    sys.exit(main())

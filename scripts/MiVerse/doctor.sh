#!/usr/bin/env bash
# Preflight checks for MiVerse eval in the shared RoboDojo environment. Does not launch Isaac Sim or a policy server.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

policy_dir=""
dataset="MiVerse"
env_cfg=""
task_name=""
ckpt_name=""
sim_env="RoboDojo"
policy_env=""
skip_isaac="false"
skip_conda="false"
skip_policy="false"
summary_path=""

usage() {
  cat <<'EOF'
Usage: bash scripts/MiVerse/doctor.sh [options]

Options:
  --dataset NAME        MiVerse only (default: MiVerse)
  --policy-dir PATH     Optional XPolicyLab policy directory to validate
  --env-cfg NAME        MiVerse env_cfg stem (default: miverse)
  --task NAME           Task (default: place_bread_in_oven)
  --ckpt NAME           Optional checkpoint directory under policy checkpoints
  --sim-env NAME        Simulator conda env for Isaac imports (default: RoboDojo)
  --policy-env NAME     Optional policy conda env to check
  --summary PATH        Write JSON summary to PATH
  --skip-isaac          Skip isaacsim/isaaclab import check
  --skip-conda          Skip conda env existence checks
  --skip-policy         Skip policy-dir/deploy/checkpoint checks
  -h, --help            Show this help
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[miverse doctor] Missing value for $1" >&2
    exit 2
  fi
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
    --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --task) need_value "$@"; task_name="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt_name="$2"; shift 2 ;;
    --sim-env) need_value "$@"; sim_env="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --summary) need_value "$@"; summary_path="$2"; shift 2 ;;
    --skip-isaac) skip_isaac="true"; shift ;;
    --skip-conda) skip_conda="true"; shift ;;
    --skip-policy) skip_policy="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[miverse doctor] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

profile_args=()
[[ -z "${dataset}" ]] || profile_args+=(--dataset "${dataset}")
[[ -z "${env_cfg}" ]] || profile_args+=(--env-cfg "${env_cfg}")
profile="$(python3 "${ROOT_DIR}/scripts/MiVerse/benchmark_profile.py" "${profile_args[@]}")"
IFS=$'\t' read -r dataset env_cfg config_name <<< "${profile}"
task_name="${task_name:-place_bread_in_oven}"

RESULTS_FILE="$(mktemp)"
INVENTORY_FILE="$(mktemp)"
trap 'rm -f "${RESULTS_FILE}" "${INVENTORY_FILE}"' EXIT

record() {
  local status="$1"
  local name="$2"
  local message="$3"
  printf '[%s] %s - %s\n' "${status}" "${name}" "${message}"
  printf '%s\t%s\t%s\n' "${status}" "${name}" "${message}" >> "${RESULTS_FILE}"
}

check_path() {
  local kind="$1"
  local path="$2"
  local name="$3"
  if [[ "${kind}" == "dir" && -d "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  elif [[ "${kind}" == "file" && -f "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  else
    record "FAIL" "${name}" "missing ${kind}: ${path}"
  fi
}

echo "[miverse doctor] root=${ROOT_DIR}"
echo "[miverse doctor] dataset=${dataset} env_cfg=${env_cfg} task=${task_name}"
if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  echo "[miverse doctor] policy_dir=${policy_dir}"
else
  echo "[miverse doctor] policy checks disabled"
fi

check_path dir "${ROOT_DIR}/env" "env package"
check_path file "${ROOT_DIR}/scripts/MiVerse/miverse.sh" "MiVerse launcher"
check_path file "${ROOT_DIR}/scripts/MiVerse/task_inventory.py" "task inventory helper"
check_path file "${ROOT_DIR}/env_cfg/${env_cfg}.yml" "env_cfg"
check_path file "${ROOT_DIR}/task/${dataset}/config/${task_name}.yml" "task config"
check_path dir "${ROOT_DIR}/Assets/Robots" "robot assets"
check_path dir "${ROOT_DIR}/Assets/Object/${dataset}" "object assets"
check_path dir "${ROOT_DIR}/Assets/Eval_Layout/${dataset}/${config_name}" "eval layouts"
check_path dir "${ROOT_DIR}/Assets/Material" "materials"

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  check_path file "${policy_dir}/eval.sh" "policy eval.sh"
  check_path file "${policy_dir}/deploy.yml" "policy deploy.yml"
  if [[ -n "${ckpt_name}" ]]; then
    check_path dir "${policy_dir}/checkpoints/${ckpt_name}" "policy checkpoint"
  else
    record "WARN" "policy checkpoint" "skipped; pass --ckpt to validate"
  fi
else
  record "WARN" "policy checks" "skipped; pass --policy-dir to validate a policy"
fi

if python3 "${ROOT_DIR}/scripts/MiVerse/task_inventory.py" --dataset "${dataset}" --format json --check >"${INVENTORY_FILE}"; then
  task_counts="$(python3 - "${INVENTORY_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding='utf-8') as f:
    payload = json.load(f)
print(payload['counts'])
PY
)"
  record "PASS" "task inventory" "${task_counts}"
else
  record "FAIL" "task inventory" "task inventory check failed"
fi

if python3 - "${ROOT_DIR}" "${env_cfg}" "${dataset}" <<'PY'; then
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
env_cfg = sys.argv[2]
dataset = sys.argv[3]
cfg_path = root / "env_cfg" / f"{env_cfg}.yml"
cfg = yaml.safe_load(cfg_path.read_text()) or {}
if cfg.get("benchmark") != dataset:
    raise SystemExit(f"env_cfg {env_cfg!r} uses benchmark {cfg.get('benchmark')!r}, not {dataset!r}")
refs = cfg.get("config", {})
required = [
    root / "env_cfg" / "sim" / f"{refs.get('sim')}.yml",
    root / "env_cfg" / "scene" / f"{refs.get('scene')}.yml",
    root / "env_cfg" / "robot" / f"{refs.get('robot')}.yml",
    root / "env_cfg" / "camera" / f"{refs.get('camera')}.yml",
    root / "env_cfg" / "robot" / "_robot_info.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("\n".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
  record "PASS" "env_cfg references" "benchmark matches; all referenced sim/scene/robot/camera files exist"
else
  record "FAIL" "env_cfg references" "invalid benchmark or missing referenced env_cfg file"
fi

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
if python3 - "${policy_dir}" <<'PY'; then
from pathlib import Path
import sys
import yaml

policy_dir = Path(sys.argv[1])
cfg = yaml.safe_load((policy_dir / "deploy.yml").read_text()) or {}
processor = cfg.get("processor_path")
if processor:
    path = Path(processor).expanduser()
    if not path.is_absolute():
        path = policy_dir / path
    if not path.exists():
        raise SystemExit(f"missing processor_path: {path}")
PY
  record "PASS" "policy processor" "deploy.yml processor_path exists"
else
  record "FAIL" "policy processor" "deploy.yml processor_path is missing"
fi
else
  record "WARN" "policy processor" "skipped; pass --policy-dir to validate"
fi

if python3 - "${ROOT_DIR}" "${dataset}" <<'PY'; then
import sys
sys.path.insert(0, sys.argv[1])
from env.benchmark import BenchmarkPaths
assert BenchmarkPaths(sys.argv[2]).config_dir.is_dir()
PY
  record "PASS" "python import" "env.benchmark imports and resolves ${dataset}"
else
  record "FAIL" "python import" "cannot import env.benchmark or resolve ${dataset}"
fi

if [[ "${skip_conda}" == "true" ]]; then
  record "WARN" "conda envs" "skipped by --skip-conda"
elif command -v conda >/dev/null 2>&1; then
  envs="$(conda env list | awk 'NF && $1 !~ /^#/ {print $1}')"
  if grep -qx "${sim_env}" <<< "${envs}"; then
    record "PASS" "sim conda env" "${sim_env}"
  else
    record "FAIL" "sim conda env" "${sim_env} not found"
  fi
  if [[ -z "${policy_env}" ]]; then
    record "WARN" "policy conda env" "skipped; pass --policy-env to validate"
  elif [[ "${policy_env}" == "uv" || "${policy_env}" == */* ]]; then
    record "WARN" "policy conda env" "policy env is path/uv; conda check skipped"
  elif grep -qx "${policy_env}" <<< "${envs}"; then
    record "PASS" "policy conda env" "${policy_env}"
  else
    record "FAIL" "policy conda env" "${policy_env} not found"
  fi
else
  record "FAIL" "conda" "conda command not found"
fi

if [[ "${skip_isaac}" == "true" ]]; then
  record "WARN" "Isaac imports" "skipped by --skip-isaac"
elif command -v conda >/dev/null 2>&1; then
  if conda run -n "${sim_env}" python - <<'PY'; then
import isaacsim  # noqa: F401
import isaaclab  # noqa: F401
PY
    record "PASS" "Isaac imports" "isaacsim and isaaclab import in ${sim_env}"
  else
    record "FAIL" "Isaac imports" "isaacsim/isaaclab import failed in ${sim_env}"
  fi
else
  record "FAIL" "Isaac imports" "conda unavailable; cannot check ${sim_env}"
fi

if [[ -n "${summary_path}" ]]; then
  mkdir -p "$(dirname "${summary_path}")"
  python3 - "${RESULTS_FILE}" "${summary_path}" <<'PY'
import json
from pathlib import Path
import sys

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    status, name, message = line.split("\t", 2)
    rows.append({"status": status, "name": name, "message": message})
payload = {
    "counts": {
        "pass": sum(row["status"] == "PASS" for row in rows),
        "warn": sum(row["status"] == "WARN" for row in rows),
        "fail": sum(row["status"] == "FAIL" for row in rows),
    },
    "checks": rows,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  echo "[miverse doctor] summary=${summary_path}"
fi

fail_count="$(awk -F '\t' '$1 == "FAIL" {count++} END {print count + 0}' "${RESULTS_FILE}")"
warn_count="$(awk -F '\t' '$1 == "WARN" {count++} END {print count + 0}' "${RESULTS_FILE}")"
pass_count="$(awk -F '\t' '$1 == "PASS" {count++} END {print count + 0}' "${RESULTS_FILE}")"
echo "[miverse doctor] pass=${pass_count} warn=${warn_count} fail=${fail_count}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi

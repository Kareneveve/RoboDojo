#!/usr/bin/env bash
# MiVerse CLI; the original RoboDojo entry points remain unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash scripts/MiVerse/miverse.sh <command> [options]

Commands:
  tasks       List the eight MiVerse tasks
  doctor      Check MiVerse assets/configs/environment
  eval        Evaluate one task with a fresh policy server
  server      Start a policy server; omit --task for batch server mode
  client      Connect to external policy servers; omit --task for a task sweep
  smoke       Evaluate selected/all tasks (default: one episode each)
  benchmark   Evaluate selected/all tasks with --eval-num NUM|native
  summarize   Summarize MiVerse results, keeping different experiments separate

Defaults: dataset MiVerse, env profile miverse, simulator environment RoboDojo.
Run a command with --help for its options. See scripts/MiVerse/README.md.
EOF
}

single_help() {
  cat <<EOF
Usage: bash scripts/MiVerse/miverse.sh ${command} [options]

eval / server:
  --policy-dir PATH     XPolicyLab policy directory
  --task TASK           MiVerse task (required for eval)
  --ckpt NAME           Checkpoint name
  --policy-env ENV      Policy environment

client:
  --policy-name NAME    XPolicyLab policy name (or use --policy-dir PATH)
  --policy-host HOST    Policy server host (default: 127.0.0.1)
  --policy-port PORT    Policy server port
  --task TASK           Omit to evaluate selected/all tasks in batch mode

Common:
  --env-cfg NAME        MiVerse profile filename stem (default: miverse)
  --action-type NAME    Action interface (default: ee)
  --seed NUM            Layout seed (default: 0)
  --eval-num NUM|native Episode count for eval/client; native uses _task.yml
  --policy-gpu ID       Policy GPU for eval/server (default: 0)
  --env-gpu ID          Simulation GPU for eval/client (default: 0)
  --eval-env ENV        Simulation Conda environment for eval (default: RoboDojo)
  --bind-host HOST      Server bind address (default: 0.0.0.0)
  --dry-run            Print resolved commands without launching a policy or simulation

Batch client/server also accept --only, --tasks-file, --gpu-ids, --limit.
Use 'bash scripts/MiVerse/batch_eval.sh --help' for all batch options.
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[miverse] Missing value for $1" >&2
    exit 2
  fi
}

run_single_or_batch() {
  local dataset="MiVerse" env_cfg="miverse" result_config task="" resolved
  local args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
      --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
      --task) need_value "$@"; task="$2"; args+=("$1" "$2"); shift 2 ;;
      --eval-num)
        need_value "$@"
        if [[ "$2" == native ]]; then
          unset EVAL_NUM
        elif [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
          echo "[miverse] --eval-num must be a positive integer or native" >&2
          exit 2
        fi
        args+=("$1" "$2")
        shift 2
        ;;
      *) args+=("$1"); shift ;;
    esac
  done
  resolved="$(python3 "${SCRIPT_DIR}/benchmark_profile.py" --dataset "${dataset}" --env-cfg "${env_cfg}")"
  IFS=$'\t' read -r dataset env_cfg result_config <<< "${resolved}"
  if [[ -z "${task}" && "${command}" != eval ]]; then
    if [[ "${command}" == client ]]; then
      args=(--policy-host 127.0.0.1 "${args[@]}")
    fi
    exec bash "${SCRIPT_DIR}/batch_eval.sh" --mode "${command}" \
      --dataset "${dataset}" --env-cfg "${env_cfg}" "${args[@]}"
  fi
  if [[ -n "${task}" ]]; then
    local task_names
    task_names="$(python3 "${SCRIPT_DIR}/task_inventory.py" --only-runnable)"
    if ! grep -Fxq -- "${task}" <<< "${task_names}"; then
      echo "[miverse] Unknown MiVerse task: ${task}" >&2
      exit 2
    fi
  fi
  exec bash "${ROOT_DIR}/scripts/robodojo.sh" "${command}" \
    --dataset "${dataset}" --env-cfg "${env_cfg}" "${args[@]}"
}

[[ $# -gt 0 ]] || { usage; exit 2; }
command="$1"
shift

case "${command}" in
  tasks) exec python3 "${SCRIPT_DIR}/task_inventory.py" "$@" ;;
  doctor) exec bash "${SCRIPT_DIR}/doctor.sh" "$@" ;;
  summarize) exec python3 "${SCRIPT_DIR}/summarize_result.py" "$@" ;;
  smoke|benchmark)
    for arg in "$@"; do
      if [[ "$arg" == -h || "$arg" == --help ]]; then
        exec bash "${SCRIPT_DIR}/batch_eval.sh" --help
      fi
    done
    if [[ "${command}" == smoke ]]; then
      exec bash "${SCRIPT_DIR}/batch_eval.sh" --eval-num "${EVAL_NUM:-1}" "$@"
    fi
    has_eval_num="false"
    for arg in "$@"; do
      [[ "$arg" != --eval-num ]] || has_eval_num="true"
    done
    if [[ "${has_eval_num}" != true && -z "${EVAL_NUM:-}" ]]; then
      echo "[miverse benchmark] Pass --eval-num NUM|native or set EVAL_NUM." >&2
      exit 2
    fi
    exec bash "${SCRIPT_DIR}/batch_eval.sh" "$@"
    ;;
  eval|server|client)
    for arg in "$@"; do
      if [[ "$arg" == -h || "$arg" == --help ]]; then
        single_help
        exit 0
      fi
    done
    run_single_or_batch "$@"
    ;;
  -h|--help) usage ;;
  *) echo "[miverse] Unknown command: ${command}" >&2; usage >&2; exit 2 ;;
esac

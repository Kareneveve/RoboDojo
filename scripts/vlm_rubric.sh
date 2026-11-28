#!/usr/bin/env bash
# Unified entry point for RoboDojo VLM-as-a-Judge Rubric evaluation,
# structured log management, and safe `_result.json` backup / apply / restore.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_LOG_DIR="${VLM_RUBRIC_LOG_DIR:-${ROOT_DIR}/logs/vlm_rubric}"
DEFAULT_CONDA_ENV="${VLM_CONDA_ENV:-RoboDojo}"

usage() {
  cat <<'EOF'
Usage: bash scripts/vlm_rubric.sh <command> [options]

Commands:
  eval     Batch-evaluate eval_result runs against VLM Rubrics and safely update _result.json
  judge    Grade a single RoboDojo run directory (--run-dir <dir>)
  status   Show grading status, sim vs rubric success rates, and backup coverage (no API calls)
  apply    Apply existing rubric_eval.json into _result.json (--run-dir or --eval-root)
  restore  Restore _result.json back to original simulator values (--run-dir or --eval-root)
  agp      Batch-evaluate AGP showcase runs
  rubrics  List all tasks with built-in VLM rubrics and their criterion IDs

Common examples:
  # 1. List configured rubrics and criteria
  bash scripts/vlm_rubric.sh rubrics

  # 2. Inspect runs and grading/backup status without calling any VLM API
  bash scripts/vlm_rubric.sh status --eval-root eval_result

  # 3. Batch-evaluate all rubric tasks (only sim_success episodes, 8 workers, skip already graded)
  bash scripts/vlm_rubric.sh eval --eval-root eval_result --only-sim-success --skip-graded --workers 8

  # 4. Judge a single run directory (auto-backs up _result.json -> _result.sim_backup.json)
  bash scripts/vlm_rubric.sh judge --run-dir eval_result/RoboDojo/push_T/.../run_0 --task push_T

  # 5. Preview or execute rollback to original simulator _result.json
  bash scripts/vlm_rubric.sh restore --eval-root eval_result --dry-run
  bash scripts/vlm_rubric.sh restore --eval-root eval_result

  # 6. Follow latest log
  tail -f logs/vlm_rubric/latest.log

Environment variables:
  VLM_JUDGE_API_KEY / DASHSCOPE_API_KEY / QWEN_VLM_API_KEY / OPENAI_API_KEY  API key (or repo-root .env)
  VLM_JUDGE_MODEL     Override default judge model (default: qwen3.8-max)
  VLM_JUDGE_BASE_URL  Override OpenAI-compatible base URL
  VLM_RUBRIC_LOG_DIR  Override log root (default: <repo>/logs/vlm_rubric)
  VLM_CONDA_ENV       Conda environment with Pillow installed (default: RoboDojo)
  PYTHON_BIN          Explicit Python binary override

Run `bash scripts/vlm_rubric.sh <command> --help` for full subcommand options.
EOF
}

load_dotenv() {
  local env_file
  for env_file in "${ROOT_DIR}/.env" "${ROOT_DIR}/.env.local"; do
    if [[ -f "${env_file}" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "${env_file}"
      set +a
    fi
  done
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi

  local candidate
  for candidate in \
    "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}" \
    "$(command -v python3 || true)" \
    "${HOME}/miniconda3/envs/${DEFAULT_CONDA_ENV}/bin/python" \
    "${HOME}/anaconda3/envs/${DEFAULT_CONDA_ENV}/bin/python" \
    "/opt/conda/envs/${DEFAULT_CONDA_ENV}/bin/python"
  do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      if "${candidate}" -c "import PIL" >/dev/null 2>&1; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    fi
  done

  if command -v conda >/dev/null 2>&1; then
    local conda_py
    conda_py="$(conda run -n "${DEFAULT_CONDA_ENV}" which python 2>/dev/null | tail -n 1 || true)"
    if [[ -n "${conda_py}" && -x "${conda_py}" ]]; then
      printf '%s\n' "${conda_py}"
      return 0
    fi
  fi

  echo "[vlm-rubric] ERROR: Could not find a Python interpreter with Pillow (PIL) installed." >&2
  echo "[vlm-rubric] Activate conda env '${DEFAULT_CONDA_ENV}' or set PYTHON_BIN=/path/to/python." >&2
  exit 1
}

has_flag() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" || "${arg}" == "${needle}="* ]]; then
      return 0
    fi
  done
  return 1
}

is_help_request() {
  has_flag "-h" "$@" || has_flag "--help" "$@"
}

require_api_key_if_needed() {
  if is_help_request "$@" || has_flag "--dry-run" "$@" || has_flag "--status" "$@" || has_flag "--list-rubrics" "$@" || has_flag "--api-key" "$@"; then
    return 0
  fi
  if [[ -z "${VLM_JUDGE_API_KEY:-}" && -z "${QWEN_VLM_API_KEY:-}" && -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[vlm-rubric] ERROR: Missing VLM Judge API key." >&2
    echo "[vlm-rubric] Set DASHSCOPE_API_KEY / VLM_JUDGE_API_KEY / QWEN_VLM_API_KEY / OPENAI_API_KEY (or add to ${ROOT_DIR}/.env), or pass --api-key." >&2
    exit 2
  fi
}

update_latest_link() {
  local target_log="$1"
  mkdir -p "${DEFAULT_LOG_DIR}"
  ln -sfn "${target_log}" "${DEFAULT_LOG_DIR}/latest.log" 2>/dev/null || true
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

cmd="$1"
shift

case "${cmd}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

load_dotenv
PYTHON_CMD="$(resolve_python)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

case "${cmd}" in
  rubrics|tasks)
    exec "${PYTHON_CMD}" "${ROOT_DIR}/scripts/vlm_rubric_eval.py" --list-rubrics
    ;;

  status)
    exec "${PYTHON_CMD}" "${ROOT_DIR}/scripts/vlm_rubric_eval.py" --dry-run "$@"
    ;;

  eval)
    require_api_key_if_needed "$@"
    extra_args=()
    if ! is_help_request "$@" && ! has_flag "--dry-run" "$@" && ! has_flag "--status" "$@"; then
      if ! has_flag "--log-dir" "$@" && ! has_flag "--log-file" "$@"; then
        run_log_dir="${DEFAULT_LOG_DIR}/${TIMESTAMP}_eval"
        mkdir -p "${run_log_dir}"
        extra_args+=(--log-file "${run_log_dir}/run.log")
        if ! has_flag "--summary-json" "$@"; then
          extra_args+=(--summary-json "${run_log_dir}/rubric_eval_summary.json")
        fi
        if ! has_flag "--summary-md" "$@"; then
          extra_args+=(--summary-md "${run_log_dir}/rubric_eval_summary.md")
        fi
        if ! has_flag "--change-log" "$@"; then
          extra_args+=(--change-log "${run_log_dir}/result_changes.jsonl")
        fi
        update_latest_link "${run_log_dir}/run.log"
      fi
      # Guard _result.json by default during batch evaluation so transient API errors never corrupt scores.
      if ! has_flag "--skip-update-on-error" "$@" && ! has_flag "--allow-update-on-error" "$@"; then
        extra_args+=(--skip-update-on-error)
      fi
    fi
    filtered_args=()
    for arg in "$@"; do
      if [[ "${arg}" != "--allow-update-on-error" ]]; then
        filtered_args+=("${arg}")
      fi
    done
    exec "${PYTHON_CMD}" "${ROOT_DIR}/scripts/vlm_rubric_eval.py" "${extra_args[@]}" "${filtered_args[@]}"
    ;;

  judge)
    require_api_key_if_needed "$@"
    extra_args=()
    if ! is_help_request "$@" && ! has_flag "--log-dir" "$@" && ! has_flag "--log-file" "$@"; then
      mkdir -p "${DEFAULT_LOG_DIR}"
      extra_args+=(--log-dir "${DEFAULT_LOG_DIR}")
    fi
    exec "${PYTHON_CMD}" "${ROOT_DIR}/scripts/vlm_rubric_judge.py" "${extra_args[@]}" "$@"
    ;;

  apply)
    extra_args=()
    if ! is_help_request "$@" && ! has_flag "--dry-run" "$@" && ! has_flag "--log-dir" "$@" && ! has_flag "--log-file" "$@"; then
      mkdir -p "${DEFAULT_LOG_DIR}"
      extra_args+=(--log-dir "${DEFAULT_LOG_DIR}")
    fi
    exec "${PYTHON_CMD}" "${ROOT_DIR}/scripts/vlm_rubric_apply.py" "${extra_args[@]}" "$@"
    ;;

  restore)
    extra_args=(--restore)
    if ! is_help_request "$@" && ! has_flag "--dry-run" "$@" && ! has_flag "--log-dir" "$@" && ! has_flag "--log-file" "$@"; then
      mkdir -p "${DEFAULT_LOG_DIR}"
      extra_args+=(--log-dir "${DEFAULT_LOG_DIR}")
    fi
    exec "${PYTHON_CMD}" "${ROOT_DIR}/scripts/vlm_rubric_apply.py" "${extra_args[@]}" "$@"
    ;;

  agp)
    require_api_key_if_needed "$@"
    extra_args=()
    if ! is_help_request "$@" && ! has_flag "--log-dir" "$@" && ! has_flag "--log-file" "$@"; then
      mkdir -p "${DEFAULT_LOG_DIR}"
      extra_args+=(--log-dir "${DEFAULT_LOG_DIR}")
    fi
    exec "${PYTHON_CMD}" "${ROOT_DIR}/scripts/vlm_rubric_agp.py" "${extra_args[@]}" "$@"
    ;;

  *)
    echo "[vlm-rubric] Unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac

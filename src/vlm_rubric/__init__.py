"""API-based VLM-as-a-Judge Rubric evaluation for RoboDojo episodes."""

from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, VLMJudgeClient, VLMJudgeError
from .judge import evaluate_run, extract_criterion_ids
from .logging_utils import configure_rubric_logging, format_duration
from .result_store import (
    RUBRIC_REPORT_JSON,
    RUBRIC_REPORT_MD,
    SIM_BACKUP_FILENAME,
    ResultStoreError,
    apply_rubric_verdict,
    backup_sim_result,
    is_rubric_graded_result,
    restore_sim_result,
)
from .rubrics import (
    GENERIC_RUBRIC,
    TASK_RUBRICS,
    configured_rubric_tasks,
    get_rubric_criteria_prompt,
    get_task_rubric,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "GENERIC_RUBRIC",
    "RUBRIC_REPORT_JSON",
    "RUBRIC_REPORT_MD",
    "ResultStoreError",
    "SIM_BACKUP_FILENAME",
    "TASK_RUBRICS",
    "VLMJudgeClient",
    "VLMJudgeError",
    "apply_rubric_verdict",
    "backup_sim_result",
    "configure_rubric_logging",
    "configured_rubric_tasks",
    "evaluate_run",
    "extract_criterion_ids",
    "format_duration",
    "get_rubric_criteria_prompt",
    "get_task_rubric",
    "is_rubric_graded_result",
    "restore_sim_result",
]

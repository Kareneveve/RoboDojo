# RoboDojo VLM-as-a-Judge Rubric 评测

RoboDojo 规则奖励负责给出 simulator 的原始 `success` 和 `score`（数量、阈值、精确位置与最终几何状态）。评测结束后，VLM-as-a-Judge Rubric 评测按任务级 **Rubric Criteria** 审查保存的三视角视频，检验过程合规性与 shortcut 违规，并在备份原始 `_result.json` 后安全写回最终裁决。

## 一键脚本入口：`scripts/vlm_rubric.sh`

脚本自动检测带 `Pillow` 的 Python/Conda 环境（默认优先 `RoboDojo`，可用 `PYTHON_BIN` / `VLM_CONDA_ENV` 覆盖），自动读取 `.env` / `.env.local`，并统一管理日志、汇总报告与 `_result.json` 备份回滚。

```bash
export DASHSCOPE_API_KEY="<your-qwen-api-key>"

# 1. 查看所有内置 Rubric 任务及 Criteria IDs
bash scripts/vlm_rubric.sh rubrics

# 2. 查看 eval_result 下待评/已评/备份状态（不调用 API，不改文件）
bash scripts/vlm_rubric.sh status --eval-root eval_result

# 3. 批量运行 VLM Rubric 评测并更新 _result.json（默认仅扫最新时间戳 + 自动备份 + 遇错保护 + 写日志与汇总）
bash scripts/vlm_rubric.sh eval \
  --eval-root eval_result \
  --only-sim-success \
  --skip-graded \
  --workers 8

# 如需包含历史旧时间戳目录，显式加 --all-runs
bash scripts/vlm_rubric.sh eval --eval-root eval_result --all-runs --only-sim-success

# 4. 对单个 run 目录执行 VLM Rubric Judge
bash scripts/vlm_rubric.sh judge \
  --run-dir eval_result/RoboDojo/put_bottles_into_dustbin/<policy>/<config>/<seed>_<info>/<run_id> \
  --task put_bottles_into_dustbin

# 5. 将已有的 rubric_eval.json 单独应用到 _result.json（支持单目录 --run-dir 或批量 --eval-root）
bash scripts/vlm_rubric.sh apply --run-dir <run_dir> --dry-run
bash scripts/vlm_rubric.sh apply --run-dir <run_dir>

# 6. 一键回滚 _result.json 为原始 simulator 结果（优先使用 _result.sim_backup.json）
bash scripts/vlm_rubric.sh restore --eval-root eval_result --dry-run
bash scripts/vlm_rubric.sh restore --eval-root eval_result
```

同样支持通过 `scripts/robodojo.sh` 调用：`rubric-eval`、`rubric-judge`、`rubric-status`、`rubric-apply`、`rubric-restore`、`rubric-agp`。

## Judge API 配置

默认使用 OpenAI-compatible Chat Completions API（默认 DashScope `https://dashscope.aliyuncs.com/compatible-mode/v1`，默认 Judge 模型 `qwen3.8-max`）：

- API Key 优先级：`--api-key` > `VLM_JUDGE_API_KEY` > `QWEN_VLM_API_KEY` > `DASHSCOPE_API_KEY` > `OPENAI_API_KEY` > 仓库根目录 `.env` / `.env.local`。
- 模型与地址覆盖：`--model` / `VLM_JUDGE_MODEL` / `QWEN_VLM_MODEL`、`--base-url` / `VLM_JUDGE_BASE_URL` / `DASHSCOPE_BASE_URL` / `OPENAI_BASE_URL`。

## 日志与报告输出管理

默认日志根目录为 `logs/vlm_rubric/`（受 `.gitignore` 中 `/logs/` 忽略，可用 `VLM_RUBRIC_LOG_DIR` 或 `--log-dir` / `--log-file` 覆盖）：

- **批量评测 (`eval`)**：每次运行创建 `logs/vlm_rubric/<YYYYMMDD_HHMMSS>_eval/`：
  - `run.log`：完整时间戳日志（每条 episode 开始/完成/耗时/verdict 与 `_result.json` 改动明细）
  - `rubric_eval_summary.json` / `rubric_eval_summary.md`：全部 run 的汇总表（含 `Sim SR -> Final SR` 与翻盘数）
  - `result_changes.jsonl`：逐条记录每次 `_result.json` 变更的 JSONL 审计日志
- **单次命令 (`judge` / `apply` / `restore` / `agp`)**：写入 `logs/vlm_rubric/<YYYYMMDD_HHMMSS>_<cmd>.log`。
- **实时追踪**：`logs/vlm_rubric/latest.log` 始终指向最新日志文件，可直接 `tail -f logs/vlm_rubric/latest.log`。
- **单 run 报告**：默认在对应 run 目录（或 `--output-dir`）生成 `rubric_eval.json` 和 `rubric_eval.md`。

## Rubric 输出 Schema（对齐 LLM Rubric-as-a-Judge 规范）

每条 episode 的 Judge 输出统一采用标准 Rubric 字段：
- `verdict`：`"pass" | "fail" | "uncertain"`
- `confidence`：`0.0 ~ 1.0`
- `violation_detected`：`bool`（是否检测到作弊/违规 shortcut）
- `criteria`：`[{"id": "<criterion_id>", "met": true/false, "confidence": 0.0, "evidence": [...]}]`
- `judge_status`：`"completed" | "skipped" | "error"`

## 目录与文件选择规则（对齐 `scripts/internal/summarize_result.py`）

- 批量命令（`eval` / `status` / `apply`）默认采用 **`latest_only`** 模式：按 `(task, policy, embodiment, seed)` 分组，仅选取**最新时间戳目录**（目录名按字典序最大且包含合法 `_result.json`），自动跳过历史旧运行目录，确保与 `summarize_result.py` 统计的目录完全一致。
- 如需回溯所有历史时间戳目录，可显式传入 `--all-runs`。
- `restore` 默认扫描全部时间戳目录，确保不会遗漏任何此前修改过的 `_result.json`。

## `_result.json` 修改与安全回滚机制

1. **首次自动备份**：原地更新 `_result.json` 前自动保存原始 simulator 结果至同目录 `_result.sim_backup.json`（仅首次写入，后续重评不会覆盖原始备份；传 `--no-backup` 可关闭）。
2. **遇错保护**：`scripts/vlm_rubric.sh eval` 默认启用 `--skip-update-on-error`，当某 run 存在 API/视频错误时跳过覆盖 `_result.json`（显式放行传 `--allow-update-on-error`）。
3. **非原地输出**：`judge` / `apply` 支持 `--result-output` / `--output` 将结果写到独立文件（如 `_result.rubric.json`），保持原 `_result.json` 不变；传 `--no-update-result` 仅生成 `rubric_eval.json/md`。
4. **字段保留与变更审计**：
   - 每个 episode 保留 `sim_success` / `sim_score`，写入 `rubric_verdict`、`rubric_confidence`、`rubric_judge_status`、`rubric_violation_detected`、`final_status`、`final_success`、`final_score`、`final_decision_reason`，并更新 `success` / `score`。
   - 顶层保留 `sim_success_rate`、`sim_score`、`sim_eval_time`，在 `rubric_eval.changes` 中记录指标变化与翻盘 episode 列表。
5. **一键回滚 (`restore`)**：优先从 `_result.sim_backup.json` 完整恢复；若无备份文件也会基于 `sim_*` 字段干净还原（同时兼容旧版 `vlm_*` 字段）。

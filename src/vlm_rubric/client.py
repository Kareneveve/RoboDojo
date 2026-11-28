"""OpenAI-compatible multimodal Judge API client for RoboDojo VLM Rubric evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "qwen3.8-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_dotenv_values() -> dict[str, str]:
    """Read key=value pairs from repo-root .env / .env.local if present."""
    values: dict[str, str] = {}
    for env_path in (REPO_ROOT / ".env", REPO_ROOT / ".env.local"):
        if not env_path.is_file():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    values[key] = value
        except OSError:
            continue
    return values


class VLMJudgeError(RuntimeError):
    """Raised when a VLM Judge API request cannot be completed or decoded."""


class VLMJudgeClient:
    """Call an OpenAI-compatible multimodal Chat Completions API without extra SDKs.

    Defaults to Alibaba Cloud DashScope's OpenAI-compatible endpoint and
    ``qwen3.8-max``. Model, endpoint, and key can be overridden via arguments,
    environment variables (``VLM_JUDGE_API_KEY`` / ``QWEN_VLM_API_KEY`` /
    ``DASHSCOPE_API_KEY`` / ``OPENAI_API_KEY``), or repo-root ``.env`` / ``.env.local``.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1800,
        timeout_s: float = 300.0,
        retries: int = 3,
        retry_delay_s: float = 2.0,
    ) -> None:
        dotenv = _read_dotenv_values()
        self.model = (
            model
            or os.environ.get("VLM_JUDGE_MODEL")
            or os.environ.get("QWEN_VLM_MODEL")
            or dotenv.get("VLM_JUDGE_MODEL")
            or dotenv.get("QWEN_VLM_MODEL")
            or DEFAULT_MODEL
        )
        self.base_url = (
            base_url
            or os.environ.get("VLM_JUDGE_BASE_URL")
            or os.environ.get("QWEN_VLM_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or dotenv.get("VLM_JUDGE_BASE_URL")
            or dotenv.get("QWEN_VLM_BASE_URL")
            or dotenv.get("DASHSCOPE_BASE_URL")
            or dotenv.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("VLM_JUDGE_API_KEY")
            or os.environ.get("QWEN_VLM_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or dotenv.get("VLM_JUDGE_API_KEY")
            or dotenv.get("QWEN_VLM_API_KEY")
            or dotenv.get("DASHSCOPE_API_KEY")
            or dotenv.get("OPENAI_API_KEY")
        )
        self.max_tokens = max(1, int(max_tokens))
        self.timeout_s = float(timeout_s)
        self.retries = max(1, int(retries))
        self.retry_delay_s = max(0.0, float(retry_delay_s))

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(self, content: list[dict[str, Any]], *, system_prompt: str) -> str:
        """Submit one multimodal judge request and return its text content."""

        if not self.api_key:
            raise VLMJudgeError(
                "Missing VLM Judge API key: set DASHSCOPE_API_KEY, VLM_JUDGE_API_KEY, "
                "QWEN_VLM_API_KEY, or OPENAI_API_KEY, or pass --api-key."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        if "json" in system_prompt.lower():
            payload["response_format"] = {"type": "json_object"}
        request = Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    body = response.read().decode("utf-8")
                return self._extract_content(json.loads(body))
            except HTTPError as exc:
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    error_body = ""
                detail = f"HTTP {exc.code} {exc.reason}"
                if error_body:
                    detail = f"{detail}: {error_body[:500]}"
                last_error = VLMJudgeError(detail)
                # Do not retry non-transient 4xx client/auth/model errors.
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    raise VLMJudgeError(f"VLM Judge API request failed: {detail}") from exc
                if attempt + 1 < self.retries:
                    delay = self.retry_delay_s * (2**attempt)
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    time.sleep(delay)
            except (URLError, TimeoutError, OSError, ValueError, VLMJudgeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_delay_s * (2**attempt))
        raise VLMJudgeError(f"VLM Judge request failed after {self.retries} attempts: {last_error}") from last_error

    @staticmethod
    def _extract_content(response: Mapping[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMJudgeError("VLM Judge response has no choices[0].message.content") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping) and part.get("text") is not None
            )
        return str(content)


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "VLMJudgeClient", "VLMJudgeError"]

from __future__ import annotations

import json
import os
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from .models import Candidate, GenerationMode, Problem


class Generator(Protocol):
    def generate(
        self,
        problem: Problem,
        mode: GenerationMode,
        target_step_id: str | None,
        reflection: str | None,
    ) -> Candidate: ...


def _load_config_file() -> dict[str, Any]:
    for path in (Path("config.json"), Path(__file__).parent.parent.parent / "config.json"):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


class OpenAICompatibleClient:
    """Minimal stdlib client for OpenAI and SiliconFlow chat-completions APIs."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls, provider: str, model: str | None = None) -> "OpenAICompatibleClient":
        provider = provider.lower()
        cfg = _load_config_file()
        prov_cfg = cfg.get(provider, {}) if isinstance(cfg.get(provider), dict) else {}

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY") or prov_cfg.get("api_key")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment or config.json")
            base_url = os.getenv("OPENAI_BASE_URL") or prov_cfg.get("base_url", "https://api.openai.com/v1")
            target_model = model or os.getenv("OPENAI_MODEL") or prov_cfg.get("model", "gpt-4o-mini")
            timeout = int(os.getenv("LLM_TIMEOUT", "180"))
            return cls(base_url, api_key, target_model, timeout=timeout)

        if provider == "siliconflow":
            api_key = os.getenv("SILICONFLOW_API_KEY") or prov_cfg.get("api_key")
            if not api_key:
                raise ValueError("SILICONFLOW_API_KEY is not set in environment or config.json")
            base_url = os.getenv("SILICONFLOW_BASE_URL") or prov_cfg.get("base_url", "https://api.siliconflow.cn/v1")
            target_model = model or os.getenv("SILICONFLOW_MODEL") or prov_cfg.get("model", "deepseek-ai/DeepSeek-V3")
            timeout = int(os.getenv("LLM_TIMEOUT", "180"))
            return cls(base_url, api_key, target_model, timeout=timeout)

        raise ValueError("provider must be 'openai' or 'siliconflow'")

    def _write_audit(self, payload: dict[str, Any]) -> None:
        audit_dir = os.getenv("LLM_AUDIT_DIR")
        if not audit_dir:
            return
        path = Path(audit_dir)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        json_attempts = max(1, int(os.getenv("LLM_JSON_ATTEMPTS", "2")))
        for attempt in range(json_attempts):
            started = time.perf_counter()
            request_payload = {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
            payload = json.dumps(request_payload).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 NeuroSymbolicMathAgent/0.1",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw_body = response.read().decode("utf-8")
                    result = json.loads(raw_body)
            except Exception as exc:
                body = exc.read().decode("utf-8", errors="replace") if isinstance(exc, urllib.error.HTTPError) else ""
                self._write_audit({
                    "provider_base_url": self.base_url, "model": self.model, "attempt": attempt + 1,
                    "request": request_payload, "elapsed_seconds": time.perf_counter() - started,
                    "status": "transport_error", "error_type": type(exc).__name__,
                    "error": str(exc), "response_body": body,
                })
                if isinstance(exc, urllib.error.HTTPError):
                    raise RuntimeError(f"LLM API returned HTTP {exc.code}: {body[:500]}") from exc
                raise
            content = result["choices"][0]["message"].get("content") or ""
            self._write_audit({
                "provider_base_url": self.base_url, "model": self.model, "attempt": attempt + 1,
                "request": request_payload, "elapsed_seconds": time.perf_counter() - started,
                "status": "ok", "response": result,
            })
            try:
                return self._decode_json(content)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < json_attempts:
                    messages.extend([
                        {"role": "assistant", "content": content[-12000:]},
                        {"role": "user", "content": "The previous response was invalid or truncated JSON. Return a corrected, concise, complete JSON object only."},
                    ])
        raise ValueError(f"Model failed to return valid JSON after repair: {last_error}")
    @staticmethod
    def _decode_json(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Model output must be a JSON object.")
        return parsed


class LLMGenerator:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def generate(
        self,
        problem: Problem,
        mode: GenerationMode,
        target_step_id: str | None,
        reflection: str | None,
    ) -> Candidate:
        schema = {
            "question": "student-facing text",
            "variables": ["x"],
            "equations": ["x + 2 = 5"],
            "answer": "x = 3",
            "target_step_id": target_step_id,
            "domain_constraints": [
                {"variable": "x", "expression": "x > 0", "require_all_solutions": True}
            ],
            "metadata": {"generation_notes": "short description"},
        }
        mode_rule = (
            "Create an isomorphic variant: preserve equation count, variable count, degree, operation structure, "
            "concepts, and step dependency logic while changing surface context and coefficients."
            if mode == GenerationMode.ISOMORPHIC else
            f"Create a reduced sub-problem isolating step {target_step_id!r}; remove downstream complexity."
        )
        user = (
            f"SOURCE PROBLEM:\n{json.dumps(asdict(problem), ensure_ascii=False)}\n\n"
            f"MODE RULE:\n{mode_rule}\n\n"
            f"OUTPUT SHAPE EXAMPLE:\n{json.dumps(schema, ensure_ascii=False)}"
        )
        if reflection:
            user += f"\n\nVERIFIER REFLECTION:\n{reflection}"
        data = self.client.complete_json(
            "You generate pedagogically appropriate mathematics items. Output strict JSON only. equations must be "
            "a list of strings. Include numeric-given equations so every declared variable has a finite determined "
            "solution. Use exact integers or fractions, never decimal floats. Domain constraints are inequalities "
            "or Ne(x, 0), never assignments. Never include undeclared symbols.",
            user,
        )
        return Candidate.from_dict(data)


class SequenceGenerator:
    """Deterministic generator useful for demos and tests of the reflection loop."""

    def __init__(self, candidates: list[Candidate]) -> None:
        if not candidates:
            raise ValueError("At least one candidate is required.")
        self.candidates = candidates
        self.calls = 0
        self.reflections: list[str | None] = []

    def generate(self, problem: Problem, mode: GenerationMode, target_step_id: str | None, reflection: str | None) -> Candidate:
        self.reflections.append(reflection)
        candidate = self.candidates[min(self.calls, len(self.candidates) - 1)]
        self.calls += 1
        return candidate









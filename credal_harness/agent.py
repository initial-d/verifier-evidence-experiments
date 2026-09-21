"""Small OpenAI-compatible JSON client used by optional live experiments.

Credentials are read at runtime from ``OPENAI_COMPATIBLE_API_KEY`` or from a
user-provided file path. No credential is stored in generated result files.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests


@dataclass(frozen=True)
class ChatAPIConfig:
    endpoint: str
    model: str
    authorization_source: str | None = None
    timeout: int = 90
    max_retries: int = 4
    user: str = "source-aware-verifier-artifact"


class HostedChatClient:
    def __init__(self, config: ChatAPIConfig, cache_dir: str = "experiments/cache/hosted") -> None:
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _authorization(self) -> str:
        value = os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        if not value and self.config.authorization_source:
            value = Path(self.config.authorization_source).read_text(encoding="utf-8").strip()
        if not value:
            raise RuntimeError(
                "Set OPENAI_COMPATIBLE_API_KEY or pass --auth-source for live hosted-model reruns."
            )
        match = re.search(r"['\"]Authorization['\"]\\s*:\\s*['\"]([^'\"]+)['\"]", value)
        value = match.group(1).strip() if match else value.strip()
        if not value or "\n" in value or "\r" in value:
            raise RuntimeError("Credential source must contain one raw key or Authorization header.")
        return value if value.lower().startswith("bearer ") else f"Bearer {value}"

    @staticmethod
    def parse_json(content: str) -> Mapping[str, Any]:
        text = content.strip()
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                value = json.loads(fenced.group(1))
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass

        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                patched = re.sub(r"\\btrue\\b", "True", candidate)
                patched = re.sub(r"\\bfalse\\b", "False", patched)
                patched = re.sub(r"\\bnull\\b", "None", patched)
                value = ast.literal_eval(patched)
            if isinstance(value, dict):
                return value

        raise ValueError(f"model did not return a JSON object: {text[:240]}")

    def complete_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> Mapping[str, Any]:
        cache_payload = json.dumps(
            {"model": self.config.model, "messages": messages, "temperature": temperature},
            ensure_ascii=False,
            sort_keys=True,
        )
        key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["parsed"]

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "top_p": 1,
            "user": self.config.user,
        }
        last_error: str | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                started = time.perf_counter()
                response = self.session.post(
                    self.config.endpoint,
                    headers={"Authorization": self._authorization(), "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.config.timeout,
                )
                latency = time.perf_counter() - started
                response.raise_for_status()
                body = response.json()
                if body.get("error") or not body.get("choices"):
                    raise RuntimeError(json.dumps(body.get("error", body), ensure_ascii=False)[:500])
                content = body["choices"][0].get("message", {}).get("content", "")
                parsed = dict(self.parse_json(content))
                cache_path.write_text(
                    json.dumps(
                        {
                            "parsed": parsed,
                            "requested_model": self.config.model,
                            "returned_model": body.get("model"),
                            "latency_seconds": latency,
                            "usage": body.get("usage", {}),
                            "credentials_stored": False,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return parsed
            except Exception as exc:
                last_error = repr(exc)
                time.sleep(min(2**attempt, 12))

        raise RuntimeError(f"hosted model request failed: {last_error}")


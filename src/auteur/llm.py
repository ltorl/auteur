"""Talking to ChatGPT through the openai-oauth proxy.

github.com/EvanZhouDev/openai-oauth exposes your ChatGPT/Codex sign-in as an
OpenAI-compatible endpoint on http://127.0.0.1:10531/v1 — no API key. We start
it on demand (``npx openai-oauth --detach``) and speak plain OpenAI to it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)
PACKAGE = "openai-oauth@latest"
Progress = Callable[[int, str], None]


class ProxyError(RuntimeError):
    """The director can't be reached; ``hint`` says what to run."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def codex_auth_file() -> Path:
    home = os.environ.get("CODEX_HOME")
    return Path(home, "auth.json") if home else Path("~/.codex/auth.json").expanduser()


def proxy_up(base_url: str, timeout: float = 2.0) -> bool:
    try:
        return httpx.get(base_url.rstrip("/") + "/models", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def npx() -> str | None:
    return shutil.which("npx")


def login() -> int:
    exe = npx()
    if not exe:
        raise ProxyError("npx not found", "install Node.js 18+ (e.g. `brew install node`)")
    return subprocess.call([exe, "-y", PACKAGE, "login"])


def start_proxy(base_url: str, wait: float = 60.0) -> None:
    exe = npx()
    if not exe:
        raise ProxyError("npx not found", "install Node.js 18+ (e.g. `brew install node`)")
    u = urlparse(base_url)
    cmd = [exe, "-y", PACKAGE, "--detach", "--host", u.hostname or "127.0.0.1",
           "--port", str(u.port or 10531), "--no-open"]
    subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=wait)
    deadline = time.time() + wait
    while time.time() < deadline:
        if proxy_up(base_url):
            return
        time.sleep(0.5)
    raise ProxyError("openai-oauth did not come up", f"run `npx {PACKAGE}` yourself to see why")


def stop_proxy() -> int:
    exe = npx()
    return subprocess.call([exe, "-y", PACKAGE, "stop"]) if exe else 1


def _version(model_id: str) -> tuple[float, ...]:
    return tuple(float(x) for x in re.findall(r"\d+(?:\.\d+)?", model_id)[:2]) or (0.0,)


def rank_text_models(ids: list[str]) -> list[str]:
    """Best creative-writing model first: newest, full-size, non-codex."""
    text = [m for m in ids if "image" not in m and "embed" not in m and "tts" not in m]

    def score(m: str):
        small = any(t in m for t in ("mini", "nano", "spark"))
        return (not small, "codex" not in m, _version(m), -len(m))

    return sorted(text, key=score, reverse=True)


def rank_image_models(ids: list[str]) -> list[str]:
    return sorted([m for m in ids if "image" in m], key=_version, reverse=True)


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class LLM:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self._client = client
        self._models: list[str] | None = None
        self._json_mode = True
        self._effort = bool(settings.reasoning_effort)

    # -- connection --------------------------------------------------------- #
    def ensure(self) -> None:
        if self._client is not None:
            return
        base = self.settings.base_url
        if not proxy_up(base):
            if not self.settings.auto_proxy:
                raise ProxyError(f"no director at {base}", f"start it with `npx {PACKAGE}`")
            if not codex_auth_file().exists():
                raise ProxyError("not signed in to ChatGPT", "run `auteur login` once")
            start_proxy(base)
        from openai import OpenAI

        self._client = OpenAI(base_url=base, api_key="openai-oauth", timeout=600, max_retries=2)

    @property
    def client(self):
        self.ensure()
        return self._client

    def models(self) -> list[str]:
        if self._models is None:
            self._models = [m.id for m in self.client.models.list().data]
        return self._models

    @property
    def text_model(self) -> str:
        if self.settings.text_model:
            return self.settings.text_model
        ranked = rank_text_models(self.models())
        if not ranked:
            raise ProxyError("your ChatGPT plan exposes no text models through openai-oauth")
        return ranked[0]

    @property
    def image_model(self) -> str:
        if self.settings.image_model:
            return self.settings.image_model
        ranked = rank_image_models(self.models())
        return ranked[0] if ranked else "gpt-image-2"

    # -- calls -------------------------------------------------------------- #
    def stream_text(self, messages: list[dict], progress: Progress | None = None) -> str:
        from openai import BadRequestError

        kwargs: dict = {"model": self.text_model, "messages": messages, "stream": True}
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._effort:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort
        try:
            stream = self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            msg = str(e).lower()
            if self._json_mode and "response_format" in msg:
                self._json_mode = False
            elif self._effort and "reasoning" in msg:
                self._effort = False
            else:
                raise
            return self.stream_text(messages, progress)
        parts: list[str] = []
        n = 0
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                parts.append(delta)
                n += len(delta)
                if progress:
                    progress(n, "".join(parts[-40:])[-200:])
        return "".join(parts)

    def structured(self, system: str, user: str, schema: type[T], progress: Progress | None = None,
                   history: list[dict] | None = None, attempts: int = 3) -> T:
        contract = (
            "\n\nRespond with a single JSON object (no prose, no code fences) that validates "
            f"against this JSON Schema:\n{json.dumps(schema.model_json_schema())}"
        )
        messages = [{"role": "system", "content": system + contract}, *(history or []),
                    {"role": "user", "content": user}]
        last: Exception | None = None
        for _ in range(attempts):
            text = self.stream_text(messages, progress)
            try:
                return schema.model_validate(extract_json(text))
            except (json.JSONDecodeError, ValidationError) as e:
                last = e
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That JSON was invalid: {str(e)[:1500]}\n"
                                                 "Return the complete corrected JSON object only."},
                ]
        raise RuntimeError(f"director returned invalid JSON after {attempts} tries: {last}")

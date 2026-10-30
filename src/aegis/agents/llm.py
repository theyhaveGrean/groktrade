"""Model calls.

:class:`OpenAIChatBackend` is the live client. It reads ``OPENAI_API_KEY``
inside the process and never puts that value on a request message.
:class:`ScriptedLLM` replays canned responses for tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from aegis.errors import MissingCredentials, ResearchError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMRequest:
    agent: str
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


@dataclass
class LLMResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        """One model turn. Tool calls and token counts come back together."""


class ScriptedLLM:
    """FIFO script per agent name. Used by tests so no API key is required."""

    def __init__(self, scripts: dict[str, list[LLMResponse]]) -> None:
        self.scripts = {name: list(turns) for name, turns in scripts.items()}
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        queue = self.scripts.get(request.agent)
        if not queue:
            raise ResearchError(f"No scripted response left for {request.agent}.")
        return queue.pop(0)


class OpenAIChatBackend:
    """Chat Completions tool loop against the official OpenAI SDK.

    The key is read here, from the environment, and handed only to the SDK
    constructor. It is not stored on this object and it is not copied into
    ``messages``.
    """

    def __init__(self) -> None:
        import os

        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise MissingCredentials(
                "OPENAI_API_KEY is not set. Export it in the environment. "
                "Tests should inject ScriptedLLM instead of calling the API."
            )
        self._client = OpenAI(api_key=key)

    def complete(self, request: LLMRequest) -> LLMResponse:
        messages: list[dict[str, Any]] = [{"role": "system", "content": request.system}, *request.messages]
        kwargs: dict[str, Any] = {"model": request.model, "messages": messages}
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — SDK raises its own types
            from aegis.agents.redaction import Redactor

            message = Redactor().scrub(f"OpenAI request failed for {request.agent}: {exc}")
            raise ResearchError(message) from None
        choice = response.choices[0].message
        calls: list[ToolCall] = []
        for index, tool_call in enumerate(choice.tool_calls or []):
            function = tool_call.function
            calls.append(
                ToolCall(
                    id=tool_call.id or f"call_{index}",
                    name=function.name,
                    arguments=_parse_arguments(function.arguments),
                )
            )
        usage = response.usage
        return LLMResponse(
            tool_calls=calls,
            text=choice.content or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}

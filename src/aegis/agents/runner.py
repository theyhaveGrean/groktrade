"""One agent turn: prompt, tool calls, structured submit.

This is the thin stand-in for the OpenAI Agents SDK runner. Each agent has
its own instructions, its own tools, and a pydantic output type. The loop
stops when the model calls ``submit_output`` with a payload that validates.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from aegis.agents.audit import AuditLog
from aegis.agents.budget import RunBudget
from aegis.agents.llm import LLMClient, LLMRequest, LLMResponse, ToolCall
from aegis.agents.prompts import load_prompt
from aegis.agents.redaction import Redactor
from aegis.agents.tools import Tool, ToolContext
from aegis.errors import BudgetExceeded, HoldoutViolation, ResearchError


@dataclass(frozen=True)
class AgentSpec:
    name: str
    output_model: type[BaseModel]
    max_turns: int = 6


def run_agent(
    spec: AgentSpec,
    user_message: str,
    *,
    client: LLMClient,
    model: str,
    tools: list[Tool],
    context: ToolContext,
    budget: RunBudget,
    audit: AuditLog,
    redactor: Redactor,
    prompt_version: str,
) -> BaseModel:
    system = redactor.scrub(load_prompt(spec.name, prompt_version))
    user_message = redactor.scrub(user_message)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    schemas = [tool.schema() for tool in tools] + [_submit_schema(spec.output_model)]
    allowed = {tool.name: tool for tool in tools}
    for turn in range(spec.max_turns):
        try:
            budget.ensure_can_call()
        except BudgetExceeded:
            raise
        audit.write(
            "llm_request",
            spec.name,
            {
                "model": model,
                "turn": turn,
                "system": system,
                "messages": messages,
                "tools": [item["function"]["name"] for item in schemas],
            },
        )
        request = LLMRequest(
            agent=spec.name,
            model=model,
            system=system,
            messages=messages,
            tools=schemas,
        )
        try:
            response = client.complete(request)
        except BudgetExceeded:
            raise
        except ResearchError:
            raise
        except Exception as exc:  # noqa: BLE001 — backend failures stay scrubbed
            raise ResearchError(redactor.scrub(f"{spec.name} model call failed: {exc}")) from exc
        audit.write(
            "llm_response",
            spec.name,
            {
                "turn": turn,
                "text": response.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments} for call in response.tool_calls
                ],
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
        )
        try:
            budget.charge(response.prompt_tokens, response.completion_tokens)
        except BudgetExceeded:
            raise
        submitted = _consume_response(
            spec,
            response,
            messages,
            allowed,
            context,
            audit,
            redactor,
        )
        if submitted is not None:
            return submitted
    raise ResearchError(f"{spec.name} did not submit a structured output within {spec.max_turns} turns.")


def _consume_response(
    spec: AgentSpec,
    response: LLMResponse,
    messages: list[dict[str, Any]],
    allowed: dict[str, Tool],
    context: ToolContext,
    audit: AuditLog,
    redactor: Redactor,
) -> BaseModel | None:
    if not response.tool_calls:
        parsed = _parse_loose(spec, response.text)
        if parsed is not None:
            return parsed
        messages.append(
            {
                "role": "user",
                "content": "Call the submit_output tool with the schema. Do not answer in prose.",
            }
        )
        return None

    messages.append(
        {
            "role": "assistant",
            "content": response.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in response.tool_calls
            ],
        }
    )
    submitted: BaseModel | None = None
    for call in response.tool_calls:
        audit.write(
            "tool_call",
            spec.name,
            {"id": call.id, "name": call.name, "arguments": call.arguments},
        )
        result, parsed = _invoke(spec, call, allowed, context, redactor)
        scrubbed = redactor.scrub(result)
        audit.write("tool_result", spec.name, {"id": call.id, "name": call.name, "result": scrubbed})
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(scrubbed, default=str)})
        if parsed is not None:
            submitted = parsed
    return submitted


def _invoke(
    spec: AgentSpec,
    call: ToolCall,
    allowed: dict[str, Tool],
    context: ToolContext,
    redactor: Redactor,
) -> tuple[dict[str, Any], BaseModel | None]:
    if call.name == "submit_output":
        try:
            parsed = spec.output_model.model_validate(call.arguments)
        except ValidationError as exc:
            return {"error": "invalid_output", "message": redactor.scrub(str(exc))}, None
        return {"ok": True}, parsed
    tool = allowed.get(call.name)
    if tool is None:
        return {
            "error": "tool_not_available",
            "message": f"{call.name} is not available to {spec.name}.",
        }, None
    try:
        result = tool.handler(call.arguments, context)
    except HoldoutViolation as exc:
        return {"error": "holdout_refused", "message": redactor.scrub(str(exc))}, None
    except ResearchError as exc:
        return {"error": "tool_failed", "message": redactor.scrub(str(exc))}, None
    except Exception as exc:  # noqa: BLE001 — tool bugs must not leak into the next prompt raw
        return {"error": "tool_failed", "message": redactor.scrub(f"{type(exc).__name__}: {exc}")}, None
    if not isinstance(result, dict):
        return {"error": "tool_failed", "message": "Tool did not return an object."}, None
    return result, None


def _parse_loose(spec: AgentSpec, text: str) -> BaseModel | None:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return spec.output_model.model_validate(payload)
    except ValidationError:
        return None


def _submit_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = _inline(model.model_json_schema())
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": "submit_output",
            "description": "Submit the structured result for this step. Call this once, when the answer is ready.",
            "parameters": schema,
        },
    }


def _inline(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = str(node["$ref"]).rsplit("/", 1)[-1]
                target = defs.get(name)
                if not isinstance(target, dict):
                    return {key: walk(value) for key, value in node.items() if key != "$ref"}
                merged = walk(copy.deepcopy(target))
                extra = {key: walk(value) for key, value in node.items() if key != "$ref"}
                if isinstance(merged, dict):
                    merged.update(extra)
                    return merged
                return merged
            return {key: walk(value) for key, value in node.items() if key != "$defs"}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(schema)

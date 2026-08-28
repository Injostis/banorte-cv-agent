"""Tests del loop de tool-use. La llamada a Claude se mockea con objetos
mínimos que imitan la forma real de una respuesta del SDK.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent import EMPTY_RESPONSE_FALLBACK, _content_for_model, run_agent


@dataclass
class _FakeBlock:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass
class _FakeResponse:
    stop_reason: str
    content: list[_FakeBlock] = field(default_factory=list)


class _FakeMessages:
    def __init__(self, responses: _FakeResponse | list[_FakeResponse]) -> None:
        self._responses = [responses] if isinstance(responses, _FakeResponse) else list(responses)

    def create(self, **_kwargs: Any) -> _FakeResponse:
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: _FakeResponse | list[_FakeResponse]) -> None:
        self.messages = _FakeMessages(responses)


def test_run_agent_falls_back_when_model_returns_no_text(monkeypatch):
    fake_response = _FakeResponse(stop_reason="end_turn", content=[])
    monkeypatch.setattr("app.agent.get_anthropic_client", lambda: _FakeClient(fake_response))

    result = run_agent([{"role": "user", "content": "hola"}])

    assert result.final_text == EMPTY_RESPONSE_FALLBACK


def test_run_agent_returns_real_text_when_present(monkeypatch):
    fake_response = _FakeResponse(stop_reason="end_turn", content=[_FakeBlock(type="text", text="Hola, soy Rodrigo.")])
    monkeypatch.setattr("app.agent.get_anthropic_client", lambda: _FakeClient(fake_response))

    result = run_agent([{"role": "user", "content": "hola"}])

    assert result.final_text == "Hola, soy Rodrigo."


def test_run_agent_runs_output_guardrail_with_tool_data_and_uses_its_result(monkeypatch):
    tool_use = _FakeBlock(type="tool_use", id="call_1", name="get_summary", input={})
    responses = [
        _FakeResponse(stop_reason="tool_use", content=[tool_use]),
        _FakeResponse(stop_reason="end_turn", content=[_FakeBlock(type="text", text="Trabajo en Google.")]),
    ]
    monkeypatch.setattr("app.agent.get_anthropic_client", lambda: _FakeClient(responses))
    monkeypatch.setattr("app.agent.execute_tool", lambda name, tool_input: {"empresa": "RYMA"})

    captured: dict[str, Any] = {}

    def _fake_check_output(final_text: str, tool_data: list[str]) -> str:
        captured["final_text"] = final_text
        captured["tool_data"] = tool_data
        return "Respuesta corregida por el guardrail de salida."

    monkeypatch.setattr("app.agent.check_output", _fake_check_output)

    result = run_agent([{"role": "user", "content": "¿dónde trabajas?"}])

    assert captured["final_text"] == "Trabajo en Google."
    assert captured["tool_data"] == ['{"empresa": "RYMA"}']
    assert result.final_text == "Respuesta corregida por el guardrail de salida."


def test_run_agent_includes_prior_assistant_texts_in_grounding_context(monkeypatch):
    tool_use = _FakeBlock(type="tool_use", id="call_1", name="get_skills", input={})
    responses = [
        _FakeResponse(stop_reason="tool_use", content=[tool_use]),
        _FakeResponse(stop_reason="end_turn", content=[_FakeBlock(type="text", text="Sí, uso Python.")]),
    ]
    monkeypatch.setattr("app.agent.get_anthropic_client", lambda: _FakeClient(responses))
    monkeypatch.setattr("app.agent.execute_tool", lambda name, tool_input: {"skills": ["Python"]})

    captured: dict[str, Any] = {}

    def _fake_check_output(final_text: str, tool_data: list[str]) -> str:
        captured["tool_data"] = tool_data
        return final_text

    monkeypatch.setattr("app.agent.check_output", _fake_check_output)

    conversation = [
        {"role": "user", "content": "¿en qué trabajas?"},
        {"role": "assistant", "content": "Trabajo en RYMA con Python en un sistema multiagente."},
        {"role": "user", "content": "¿usas python?"},
    ]
    run_agent(conversation)

    assert "Trabajo en RYMA con Python en un sistema multiagente." in captured["tool_data"]
    assert '{"skills": ["Python"]}' in captured["tool_data"]


def test_run_agent_skips_output_guardrail_when_no_tool_used_this_turn(monkeypatch):
    fake_response = _FakeResponse(stop_reason="end_turn", content=[_FakeBlock(type="text", text="¡De nada!")])
    monkeypatch.setattr("app.agent.get_anthropic_client", lambda: _FakeClient(fake_response))

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("check_output no debería llamarse sin tool en este turno")

    monkeypatch.setattr("app.agent.check_output", _fail_if_called)

    conversation = [
        {"role": "user", "content": "¿dónde trabajas?"},
        {"role": "assistant", "content": "Trabajo en RYMA."},
        {"role": "user", "content": "gracias!"},
    ]
    result = run_agent(conversation)

    assert result.final_text == "¡De nada!"


def test_content_for_model_extracts_fallback_from_call_tool_result():
    output = {"content": [{"type": "text", "text": "Resumen visual."}, {"type": "resource", "resource": {}}]}

    content = json.loads(_content_for_model(output))

    assert content["resumen_visual"] == "Resumen visual."
    assert "nota" in content


def test_content_for_model_passthrough_for_normal_tools():
    output = {"total_platinos": 37, "trofeos": []}

    assert json.loads(_content_for_model(output)) == output

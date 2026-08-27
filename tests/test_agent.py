"""Tests del loop de tool-use. La llamada a Claude se mockea con objetos
mínimos que imitan la forma real de una respuesta del SDK -- no se gastan
tokens reales en cada corrida.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent import EMPTY_RESPONSE_FALLBACK, _content_for_model, run_agent


@dataclass
class _FakeBlock:
    type: str
    text: str | None = None


@dataclass
class _FakeResponse:
    stop_reason: str
    content: list[_FakeBlock] = field(default_factory=list)


class _FakeMessages:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def create(self, **_kwargs: Any) -> _FakeResponse:
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeMessages(response)


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


def test_content_for_model_extracts_fallback_from_call_tool_result():
    output = {"content": [{"type": "text", "text": "Resumen visual."}, {"type": "resource", "resource": {}}]}

    content = json.loads(_content_for_model(output))

    assert content["resumen_visual"] == "Resumen visual."
    assert "nota" in content


def test_content_for_model_passthrough_for_normal_tools():
    output = {"total_platinos": 37, "trofeos": []}

    assert json.loads(_content_for_model(output)) == output

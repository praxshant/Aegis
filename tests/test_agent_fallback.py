"""The real (un-stubbed) agent with no LLM reachable must fail with AegisAgentError — the
API maps that to a clean 503 — never with a TypeError/NameError that surfaces as a 500."""
import pytest

from backend.agent.agent import AegisAgent, AegisAgentError, _discover_model


def test_no_llm_configured_raises_clean_agent_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")  # discard port: nothing listens
    agent = AegisAgent(tools=None)
    with pytest.raises(AegisAgentError) as exc:
        agent.propose_action("CUST_002", "Assess", intelligence_context={"status": "COMPLETE"})
    msg = str(exc.value)
    assert "GEMINI_API_KEY is not set" in msg and "127.0.0.1:9" in msg


class _Model:
    def __init__(self, name, actions=("generateContent",)):
        self.name, self.supported_actions = f"models/{name}", list(actions)


class _Client:
    def __init__(self, names):
        self.models = type("M", (), {"list": lambda _self: [_Model(n) if isinstance(n, str) else _Model(*n) for n in names]})()


def test_model_discovery_prefers_listed_text_models(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert _discover_model(_Client(["gemini-2.5-flash-image", "gemini-2.5-flash"])) == "gemini-2.5-flash"
    # unknown future names: pick a text-capable flash model, never an image/tts one
    picked = _discover_model(_Client(["gemini-9-flash-image", "gemini-9-flash-preview-tts", "gemini-9-flash"]))
    assert picked == "gemini-9-flash"
    # models that can't generateContent are ignored
    assert _discover_model(_Client([("gemini-2.5-flash", ("embedContent",)), "gemini-flash-latest"])) == "gemini-flash-latest"
    monkeypatch.setenv("GEMINI_MODEL", "my-pinned-model")
    assert _discover_model(_Client(["gemini-2.5-flash"])) == "my-pinned-model"

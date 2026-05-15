"""Main-flow verification for Hermes + Hindsight auto memory.

These tests intentionally do NOT call the Hindsight client directly from the
assertion path. Instead they drive ``AIAgent.run_conversation()`` with the
Hindsight memory provider enabled, then verify that:

1. completed turns are auto-retained through the provider writer queue; and
2. the next turn receives auto-recalled context injected into the API-facing
   user message.

That is the actual Hermes session path the user cares about — not a raw
``hindsight_client`` smoke test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    """Build the minimal tool schema list accepted by ``AIAgent`` init."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_response(content: str) -> SimpleNamespace:
    """Return a minimal OpenAI-style chat completion response."""
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.fixture()
def hindsight_agent(tmp_path, monkeypatch):
    """Build an ``AIAgent`` with the real Hindsight provider wired in.

    The provider is configured via the normal Hermes config path
    (``memory.provider: hindsight``), but its async client is mocked so the
    test stays local and deterministic.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    agent_cfg = {
        "model": {"context_length": 400_000},
        "memory": {
            "provider": "hindsight",
            "memory_enabled": False,
            "user_profile_enabled": False,
            "nudge_interval": 10,
        },
        "agent": {},
    }
    provider_cfg = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
        "auto_retain": True,
        "auto_recall": True,
        "retain_async": True,
        "retain_every_n_turns": 1,
        "retain_context": "conversation between Hermes Agent and the User",
        "recall_prefetch_method": "recall",
        "recall_max_tokens": 4096,
        "recall_max_input_chars": 800,
    }

    recall_payloads = [
        SimpleNamespace(results=[SimpleNamespace(text="User likes oolong tea")]),
        SimpleNamespace(results=[SimpleNamespace(text="User likes oolong tea")]),
    ]

    async def _arecall(**kwargs):
        if recall_payloads:
            return recall_payloads.pop(0)
        return SimpleNamespace(results=[])

    async def _aretain_batch(**kwargs):
        return SimpleNamespace(ok=True)

    client = MagicMock()
    client.arecall = AsyncMock(side_effect=_arecall)
    client.aretain_batch = AsyncMock(side_effect=_aretain_batch)
    client.aclose = AsyncMock()

    def _fake_get_client(self):
        self._client = client
        return client

    def _fake_provider_config():
        return dict(provider_cfg)

    with (
        patch("hermes_cli.config.load_config", return_value=agent_cfg),
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("plugins.memory.hindsight._load_config", side_effect=_fake_provider_config),
        patch("plugins.memory.hindsight._check_api_supports_update_mode_append", return_value=False),
        patch("plugins.memory.hindsight.HindsightMemoryProvider._get_client", new=_fake_get_client),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )
        agent.client = MagicMock()
        yield agent, client

    try:
        agent.shutdown_memory_provider([])
    except Exception:
        pass


def test_hindsight_auto_retain_and_auto_recall_flow(hindsight_agent):
    """A completed turn should warm recall, and the next turn should consume it.

    This is the Hermes session contract we care about:

    turn 1 -> run_conversation() -> auto-retain + queue auto-recall
    turn 2 -> run_conversation() -> prefetch consumes recalled context and injects it
    """
    agent, client = hindsight_agent
    agent.client.chat.completions.create.side_effect = [
        _mock_response("First answer"),
        _mock_response("Second answer"),
    ]

    first = agent.run_conversation("Remember that I like oolong tea")
    provider = agent._memory_manager.get_provider("hindsight")
    assert provider is not None

    # Turn 1 retain + warm-up recall happen asynchronously inside the provider.
    provider._retain_queue.join()
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=5.0)

    second = agent.run_conversation("What drink do I like?")

    # Turn 2 also retains and queues the next recall; drain for deterministic asserts.
    provider._retain_queue.join()
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=5.0)

    assert first["final_response"] == "First answer"
    assert second["final_response"] == "Second answer"
    assert len(agent.client.chat.completions.create.call_args_list) == 2

    # Auto-retain proved through the provider's async batch-retain call.
    assert client.aretain_batch.call_count == 2
    first_retain = client.aretain_batch.call_args_list[0].kwargs
    assert first_retain["bank_id"] == "test-bank"
    retained_item = first_retain["items"][0]
    assert retained_item["context"] == "conversation between Hermes Agent and the User"
    assert "Remember that I like oolong tea" in retained_item["content"]
    assert "First answer" in retained_item["content"]

    # Auto-recall proved two ways:
    #   (1) turn 1 queued a background recall query; and
    #   (2) turn 2 consumed that result into the API-facing user message.
    assert client.arecall.call_count == 2
    assert client.arecall.call_args_list[0].kwargs["query"] == "Remember that I like oolong tea"

    first_api_messages = agent.client.chat.completions.create.call_args_list[0].kwargs["messages"]
    first_user_message = [m for m in first_api_messages if m.get("role") == "user"][-1]["content"]
    assert "<memory-context>" not in first_user_message

    second_api_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    second_user_message = [m for m in second_api_messages if m.get("role") == "user"][-1]["content"]
    assert "<memory-context>" in second_user_message
    assert "User likes oolong tea" in second_user_message
    assert "What drink do I like?" in second_user_message

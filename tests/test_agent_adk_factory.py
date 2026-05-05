import sys
import types

from cartograph.agents import adk_factory


def test_is_adk_available_false_without_google_adk(monkeypatch):
    def raise_import_error():
        raise ImportError

    monkeypatch.setattr(adk_factory, "_load_llm_agent", raise_import_error)
    assert not adk_factory.is_adk_available()


def test_build_cartograph_adk_agent_uses_sub_agents(monkeypatch):
    created = []

    class FakeLlmAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(adk_factory, "_load_llm_agent", lambda: FakeLlmAgent)
    root = adk_factory.build_cartograph_adk_agent()
    assert root.kwargs["name"] == "cartograph"
    assert [agent.kwargs["name"] for agent in root.kwargs["sub_agents"]] == [
        "mapper",
        "auditor",
        "generator",
    ]
    assert len(created) == 4


def test_module_import_does_not_require_google_adk():
    assert isinstance(sys.modules["cartograph.agents.adk_factory"], types.ModuleType)

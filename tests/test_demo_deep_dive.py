from types import SimpleNamespace

from cartograph.demo.deep_dive import _adk_eval_blocker, _precomputed_payload


class FakeReport:
    def model_dump(self, mode: str):
        assert mode == "json"
        return {
            "eval_run_before": {
                "command": ["/tmp/project/.venv/bin/adk", "eval"],
                "stdout": "",
                "stderr": "line 1\nvery long local stack trace oauth2.googleapis.com/path",
            },
            "eval_run_after": {
                "command": ["adk", "eval"],
                "stdout": "x" * 300,
                "stderr": "",
            },
        }


def test_precomputed_payload_summarizes_eval_outputs():
    payload = _precomputed_payload(FakeReport())
    assert payload["eval_run_before"]["command"][0] == "adk"
    assert payload["eval_run_before"]["stderr"] == "oauth2.googleapis.com"
    assert payload["eval_run_after"]["stdout"].endswith("...")
    assert len(payload["eval_run_after"]["stdout"]) <= 240


def test_precomputed_payload_suppresses_local_warning_paths():
    class WarningReport:
        def model_dump(self, mode: str):
            assert mode == "json"
            return {
                "eval_run_before": {
                    "command": ["/tmp/project/.venv/bin/adk", "eval"],
                    "stdout": "",
                    "stderr": "/tmp/project/.venv/lib/site-packages/pkg.py: UserWarning",
                },
                "eval_run_after": None,
            }

    payload = _precomputed_payload(WarningReport())

    assert payload["eval_run_before"]["stderr"] == "ADK runtime warnings suppressed"


def test_adk_eval_blocker_distinguishes_gemini_service_disabled():
    run = SimpleNamespace(
        pass_rate=None,
        stderr="PERMISSION_DENIED SERVICE_DISABLED generativelanguage.googleapis.com",
        stdout="",
        exit_code=1,
    )
    report = SimpleNamespace(eval_run_before=run, eval_run_after=None)

    assert _adk_eval_blocker(report) == "Gemini API is disabled or unavailable"


def test_adk_eval_blocker_distinguishes_vertex_service_disabled():
    run = SimpleNamespace(
        pass_rate=None,
        stderr="PERMISSION_DENIED SERVICE_DISABLED aiplatform.googleapis.com",
        stdout="",
        exit_code=1,
    )
    report = SimpleNamespace(eval_run_before=run, eval_run_after=None)

    assert _adk_eval_blocker(report) == "Vertex Agent Platform API is disabled or unavailable"

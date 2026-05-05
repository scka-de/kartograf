from cartograph.demo.deep_dive import _precomputed_payload


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

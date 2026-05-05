import json

from cartograph.mcp_servers.eval_io_adk import read_evalset, write_evalset


def test_eval_io_adk_round_trips_unknown_fields(tmp_path):
    path = tmp_path / "sample.evalset.json"
    payload = {
        "eval_set_id": "sample",
        "eval_cases": [
            {
                "eval_id": "case_1",
                "conversation": [{"role": "user", "content": "hello"}],
                "unknown_field": {"keep": True},
            }
        ],
    }
    path.write_text(json.dumps(payload))
    cases = read_evalset(str(path))
    assert cases[0]["content"] == "hello"
    out = tmp_path / "out.evalset.json"
    write_evalset(str(out), cases, merge=False)
    round_trip = json.loads(out.read_text())
    assert round_trip["eval_cases"][0]["unknown_field"] == {"keep": True}


def test_eval_io_adk_normalizes_fallback_content(tmp_path):
    path = tmp_path / "sample.evalset.json"
    path.write_text(json.dumps({"eval_cases": [{"eval_id": "case_1", "input": "hello"}]}))
    cases = read_evalset(str(path))
    assert cases[0]["content"] == "hello"


def test_eval_io_adk_reads_current_adk_test_json_list(tmp_path):
    path = tmp_path / "simple.test.json"
    path.write_text(json.dumps([{"query": "hello", "reference": "hi"}]))
    cases = read_evalset(str(path))
    assert cases[0]["content"] == "hello"
    assert cases[0]["id"].startswith("case_")


def test_eval_io_adk_reads_old_cli_evalset_list(tmp_path):
    path = tmp_path / "sample.evalset.json"
    path.write_text(json.dumps([{"name": "case_1", "data": [{"query": "hello"}]}]))
    cases = read_evalset(str(path))
    assert cases[0]["id"] == "case_1"
    assert cases[0]["content"] == "hello"


def test_eval_io_adk_reads_current_pydantic_conversation(tmp_path):
    path = tmp_path / "sample.evalset.json"
    path.write_text(
        json.dumps(
            {
                "eval_cases": [
                    {
                        "eval_id": "case_1",
                        "conversation": [
                            {"user_content": {"parts": [{"text": "hello"}], "role": "user"}}
                        ],
                    }
                ]
            }
        )
    )
    cases = read_evalset(str(path))
    assert cases[0]["id"] == "case_1"
    assert cases[0]["content"] == "hello"

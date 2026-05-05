from cartograph.mcp_servers.eval_io_git import read_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_jsonl(str(path), [{"id": "a"}, {"id": "b"}])
    assert read_jsonl(str(path)) == [{"id": "a"}, {"id": "b"}]

from cartograph.mcp_servers.corpus_reader_bitext import fetch_examples, get_metadata, list_intents


def test_bitext_fixture_returns_examples():
    examples = fetch_examples(limit=3)
    assert len(examples) == 3
    assert examples[0]["text"]
    assert list_intents()
    assert get_metadata()["count"] >= 3

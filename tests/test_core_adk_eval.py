from cartograph.core.adk_eval import _adk_env, parse_adk_output


def test_parse_fraction_passed():
    assert parse_adk_output("17/18 passed") == (17, 1, 17 / 18)


def test_parse_json_output():
    assert parse_adk_output('{"pass_count": 3, "fail_count": 1}') == (3, 1, 0.75)


def test_parse_pass_rate_only():
    assert parse_adk_output("pass_rate: 0.85") == (None, None, 0.85)


def test_parse_adk_tests_passed_failed_summary():
    output = """
    Eval Run Summary
    customer_service_cartograph:
      Tests passed: 2
      Tests failed: 6
    """
    assert parse_adk_output(output) == (2, 6, 0.25)


def test_parse_unknown_output_returns_nones():
    assert parse_adk_output("no summary available") == (None, None, None)


def test_adk_env_prefers_developer_api_when_api_key_available(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    env = _adk_env(str(agent_dir))
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "0"
    assert str(tmp_path) in env["PYTHONPATH"]


def test_adk_env_preserves_explicit_vertex_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    env = _adk_env(str(agent_dir))
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "1"


def test_adk_env_loads_dotenv_before_subprocess_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=test-key\n")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    env = _adk_env(str(agent_dir))

    assert env["GOOGLE_API_KEY"] == "test-key"
    assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "0"

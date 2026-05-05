from pathlib import Path


def test_spec_named_scripts_exist():
    for path in [
        "scripts/fetch_bitext.py",
        "scripts/fetch_stackexchange.py",
        "scripts/fetch_github_issues.py",
        "scripts/clone_adk_samples.py",
        "scripts/run_demo.sh",
        "scripts/pre_submission_check.sh",
    ]:
        assert Path(path).exists()

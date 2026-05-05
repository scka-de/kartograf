"""MCP tool: fetch_issues for GitHub issues with cache and fixture fallback."""

from __future__ import annotations

import os

from ._common import data_path, fixture_path, load_json, run_stdio_server, save_json

_LAST_MODE = "unknown"


def fetch_issues(owner: str, repo: str, limit: int = 500, state: str = "closed") -> list[dict]:
    global _LAST_MODE
    cache = data_path("corpora", f"github_{owner}_{repo}_{state}.json")
    cached = load_json(cache, None)
    if cached is not None:
        _LAST_MODE = "real"
        return cached[:limit]
    try:
        from github import Github  # type: ignore[import-not-found]

        token = os.getenv("GITHUB_TOKEN")
        gh = Github(token) if token else Github()
        repository = gh.get_repo(f"{owner}/{repo}")
        issues = []
        for issue in repository.get_issues(state=state):
            if issue.pull_request is not None:
                continue
            issues.append(
                {
                    "id": str(issue.number),
                    "text": f"{issue.title}\n{issue.body or ''}",
                    "labels": [label.name for label in issue.labels],
                }
            )
            if len(issues) >= limit:
                break
        save_json(cache, issues)
        _LAST_MODE = "real"
        return issues
    except Exception:
        _LAST_MODE = "mocked"
        return load_json(fixture_path("corpus_reader_github.json"), [])[:limit]


def get_mode() -> str:
    return _LAST_MODE


if __name__ == "__main__":
    run_stdio_server("corpus_reader_github", {"fetch_issues": fetch_issues, "get_mode": get_mode})

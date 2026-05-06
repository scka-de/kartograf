"""Smoke tests for the visual-first audit pipeline.

These tests exercise the full audit on the bundled customer-service ADK sample using
deterministic local embeddings (no Gemini API calls), and verify the HTML report is
well-formed.
"""

from __future__ import annotations

import json
import re

import pytest

from cartograph.density import compute_gap_field
from cartograph.evals import extract_eval_inputs
from cartograph.pipeline import SurfaceBudgetError, audit_repo
from cartograph.report import _safe_json_for_script
from cartograph.surfaces import extract_surfaces

CUSTOMER_SERVICE = "data/adk_samples/python/agents/customer-service"


def test_surface_extraction_meets_floor():
    surfaces = extract_surfaces(CUSTOMER_SERVICE)
    assert len(surfaces) >= 30
    kinds = {s.kind for s in surfaces}
    assert {"prompt", "tool"} <= kinds


def test_eval_extraction_finds_test_cases():
    evals = extract_eval_inputs(CUSTOMER_SERVICE)
    assert len(evals) >= 1
    assert all(e.text.strip() for e in evals)


def test_surface_budget_too_thin_raises(tmp_path):
    (tmp_path / "tiny.py").write_text('"""Tiny module with one docstring."""\n')
    with pytest.raises(SurfaceBudgetError):
        audit_repo(str(tmp_path), str(tmp_path / "out.html"), surface_floor=30)


def test_compute_gap_field_basic():
    import numpy as np

    surfaces = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]])
    evals = np.array([[0.0, 0.05]])
    gap = compute_gap_field(surfaces, evals)
    assert gap.gap_field.shape == (100, 100)
    assert 0.0 <= gap.coverage_score <= 1.0
    assert gap.alpha > 0


def test_safe_json_for_script_escapes_script_breakouts():
    payload = {
        "danger": "</script><script>alert(1)</script>",
        "u2028": "line break",
        "u2029": "para sep",
        "amp": "a&b",
    }
    encoded = _safe_json_for_script(payload)
    assert "</script>" not in encoded
    assert "\\u003c/script\\u003e" in encoded
    assert " " not in encoded
    assert "\\u2028" in encoded
    assert " " not in encoded
    assert "\\u2029" in encoded
    assert "&" not in encoded
    assert "\\u0026" in encoded


def test_audit_repo_end_to_end_writes_html(tmp_path):
    output = tmp_path / "report.html"
    summary = audit_repo(CUSTOMER_SERVICE, str(output))
    assert output.exists()
    assert summary["surface_count"] >= 30
    html = output.read_text()
    match = re.search(r"const DATA = (\{.*?\});\n", html, re.DOTALL)
    assert match, "html payload missing"
    payload = json.loads(match.group(1))
    assert payload["summary"]["surface_count"] == summary["surface_count"]
    assert payload["summary"]["eval_count"] == summary["eval_count"]
    assert "regions" in payload
    assert "kinds" in payload["summary"]

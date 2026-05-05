import os

import pytest

from cartograph.core import storage
from cartograph.demo.deep_dive import run_customer_service_deep_dive


@pytest.mark.e2e
def test_e2e_smoke_with_live_google_runtime():
    if os.getenv("CARTOGRAPH_RUN_E2E") != "1":
        pytest.skip("Set CARTOGRAPH_RUN_E2E=1 to run the live pre-submission smoke test")
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_GENAI_USE_VERTEXAI") != "1":
        pytest.skip("Live e2e requires GOOGLE_API_KEY or configured Vertex credentials")

    payload = run_customer_service_deep_dive(limit=30)
    report = storage.load_coverage_report(payload["audit_id"])

    assert report.coverage_score >= 0.75
    assert report.eval_run_before is not None
    assert report.eval_run_after is not None
    assert report.eval_run_before.pass_rate is not None
    assert report.eval_run_after.pass_rate is not None

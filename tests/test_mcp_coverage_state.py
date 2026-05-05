from cartograph.core import storage
from cartograph.core.models import Gap, GeneratedCase
from cartograph.mcp_servers.coverage_state import (
    get_generated_cases,
    get_uncovered_regions,
    list_audits,
)


def test_coverage_state_lists_audits_and_gaps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    storage.save_gaps("audit_test", [Gap(region_id="region_00", risk_score=0.2, rank=1)])
    assert list_audits()[0]["id"] == "audit_test"
    assert get_uncovered_regions("audit_test")[0]["region_id"] == "region_00"


def test_coverage_state_filters_generated_cases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    storage.save_generated_cases(
        "audit_test",
        [
            GeneratedCase(
                id="accepted",
                region_id="region_00",
                content="a",
                validation_status="accepted",
                accepted=True,
            ),
            GeneratedCase(
                id="rejected",
                region_id="region_00",
                content="b",
                validation_status="rejected_redundant",
                accepted=False,
            ),
        ],
    )
    assert [case["id"] for case in get_generated_cases("audit_test")] == ["accepted"]
    assert len(get_generated_cases("audit_test", accepted_only=False)) == 2

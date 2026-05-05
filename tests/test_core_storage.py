from datetime import datetime

from cartograph.core import storage
from cartograph.core.models import Decision, Gap, Region


def test_storage_reconstructs_coverage_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    storage.save_regions(
        "audit_test",
        [
            Region(
                id="region_00",
                label="Refund",
                centroid=[1.0] + [0.0] * 29,
                density=1.0,
                heterogeneity=0.1,
                member_count=3,
            )
        ],
    )
    storage.save_gaps("audit_test", [Gap(region_id="region_00", risk_score=0.1, rank=1)])
    storage.save_decision(
        "audit_test",
        Decision(step=1, timestamp=datetime.utcnow(), action="ingest", reason="test"),
    )
    storage.update_audit_summary("audit_test", 0.0, 0.0, ["warning"])
    report = storage.load_coverage_report("audit_test")
    assert report.audit_id == "audit_test"
    assert report.regions[0].centroid == [1.0] + [0.0] * 29
    assert report.gaps[0].rank == 1
    assert report.decisions[0].action == "ingest"


def test_latest_audit_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage.create_audit("audit_test", "target", "bitext")
    assert storage.latest_audit_id() == "audit_test"

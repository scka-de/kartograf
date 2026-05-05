import numpy as np
import pytest

from cartograph.core.risk import compute_risk


def test_compute_risk_uses_30d(tiny_regions, eval_embeddings_30d):
    result = compute_risk(tiny_regions[1], eval_embeddings_30d)
    assert result["risk_score"] == pytest.approx(0.1)
    assert set(result["components"]) == {"density", "heterogeneity", "distance"}


def test_compute_risk_rejects_wrong_dimensions(tiny_regions):
    bad = np.asarray([[1.0, 0.0]])
    with pytest.raises(ValueError, match="30d"):
        compute_risk(tiny_regions[0], bad)

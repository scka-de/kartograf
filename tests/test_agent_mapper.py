import numpy as np
import pytest

from cartograph.agents import mapper
from cartograph.agents.mapper import build_regions, load_corpus
from cartograph.core.math import l2_normalize


def test_mapper_builds_regions_with_30d_centroids(tiny_corpus):
    vectors = l2_normalize(
        np.asarray([[1.0, 0.1] + [0.0] * 28] * 10 + [[0.1, 1.0] + [0.0] * 28] * 10)
    )
    labels = np.asarray([0] * 10 + [1] * 10)
    regions = build_regions(tiny_corpus, vectors, labels)
    assert len(regions) == 2
    assert all(len(region.centroid) == 30 for region in regions)


def test_load_corpus_rejects_empty_adapter_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mapper.corpus_reader_bitext, "fetch_examples", lambda limit=1000: [])

    with pytest.raises(RuntimeError, match="returned no rows"):
        load_corpus("audit_test", "bitext", 10, {})

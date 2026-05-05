import numpy as np

from cartograph.agents.mapper import build_regions
from cartograph.core.math import l2_normalize


def test_mapper_builds_regions_with_30d_centroids(tiny_corpus):
    vectors = l2_normalize(
        np.asarray([[1.0, 0.1] + [0.0] * 28] * 10 + [[0.1, 1.0] + [0.0] * 28] * 10)
    )
    labels = np.asarray([0] * 10 + [1] * 10)
    regions = build_regions(tiny_corpus, vectors, labels)
    assert len(regions) == 2
    assert all(len(region.centroid) == 30 for region in regions)

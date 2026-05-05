import numpy as np

from cartograph.core.clustering import (
    cluster,
    fit_reducer,
    fit_visualization_reducer,
    transform_embeddings,
)


def test_small_fixture_cluster_produces_regions():
    labels = cluster(np.ones((9, 30)))
    assert set(labels.tolist()) == {0, 1, 2}


def test_projection_reducer_outputs_30d():
    embeddings = np.eye(6, 768)
    reducer = fit_reducer(embeddings, 30)
    reduced = transform_embeddings(reducer, embeddings)
    assert reduced.shape == (6, 30)


def test_visualization_reducer_outputs_2d():
    embeddings = np.eye(6, 768)
    reducer = fit_visualization_reducer(embeddings)
    reduced = reducer.transform(embeddings)
    assert reduced.shape == (6, 2)

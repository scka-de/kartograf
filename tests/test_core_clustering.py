import numpy as np

from cartograph.core.clustering import (
    ProjectionReducer,
    _fit_projection_reducer,
    fit_visualization_reducer,
)


def test_visualization_reducer_outputs_2d():
    embeddings = np.eye(6, 768)
    reducer = fit_visualization_reducer(embeddings)
    reduced = reducer.transform(embeddings)
    assert reduced.shape == (6, 2)


def test_projection_reducer_fallback_outputs_2d():
    embeddings = np.eye(6, 768)
    reducer = _fit_projection_reducer(embeddings, 2)
    assert isinstance(reducer, ProjectionReducer)
    reduced = reducer.transform(embeddings)
    assert reduced.shape == (6, 2)


def test_projection_reducer_handles_empty_input():
    reducer = _fit_projection_reducer(np.empty((0, 768)), 2)
    reduced = reducer.transform(np.empty((0, 768)))
    assert reduced.shape == (0, 2)

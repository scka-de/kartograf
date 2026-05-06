"""Density fields and gap detection on the 2D semantic map.

Computes:
  rho_S(x) = surface density (KDE)
  rho_E(x) = eval density (KDE, mass-balanced via alpha)
  g(x)    = max(0, rho_S - alpha * rho_E)

Then thresholds g, finds connected components, ranks by integrated mass.
Pure numpy/scipy. Falls back to a manual Gaussian KDE if scipy is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class GapRegion:
    id: str
    centroid_2d: tuple[float, float]
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    integrated_mass: float
    area_cells: int


@dataclass
class GapField:
    grid_x: NDArray[np.float64]
    grid_y: NDArray[np.float64]
    rho_surfaces: NDArray[np.float64]
    rho_evals: NDArray[np.float64]
    gap_field: NDArray[np.float64]
    threshold: float
    alpha: float
    bandwidth: float
    regions: list[GapRegion] = field(default_factory=list)
    coverage_score: float = 0.0


def compute_gap_field(
    surface_points_2d: NDArray[np.float64],
    eval_points_2d: NDArray[np.float64],
    grid_size: int = 100,
    alpha: float | None = None,
    threshold_quantile: float = 0.75,
    bandwidth: float | None = None,
    pad_fraction: float = 0.10,
) -> GapField:
    if surface_points_2d.size == 0:
        raise ValueError("compute_gap_field requires at least one surface point")

    points_for_extent = (
        surface_points_2d
        if eval_points_2d.size == 0
        else np.vstack([surface_points_2d, eval_points_2d])
    )
    x_min, y_min = points_for_extent.min(axis=0)
    x_max, y_max = points_for_extent.max(axis=0)
    span_x = max(x_max - x_min, 1e-3)
    span_y = max(y_max - y_min, 1e-3)
    x_min -= span_x * pad_fraction
    x_max += span_x * pad_fraction
    y_min -= span_y * pad_fraction
    y_max += span_y * pad_fraction

    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    if bandwidth is None:
        bandwidth = _scott_bandwidth(points_for_extent)

    rho_S = _gaussian_kde(surface_points_2d, grid_points, bandwidth)
    rho_S = rho_S.reshape(grid_size, grid_size)
    rho_S = _normalize_density(rho_S, grid_x, grid_y)

    if eval_points_2d.size == 0:
        rho_E = np.zeros_like(rho_S)
    else:
        rho_E = _gaussian_kde(eval_points_2d, grid_points, bandwidth).reshape(grid_size, grid_size)
        rho_E = _normalize_density(rho_E, grid_x, grid_y)

    if alpha is None:
        alpha = (
            float(len(surface_points_2d)) / max(1, len(eval_points_2d))
            if eval_points_2d.size
            else 1.0
        )
    raw_gap = rho_S - (alpha * rho_E if eval_points_2d.size else 0.0)
    gap = np.clip(raw_gap, 0.0, None)

    # Threshold is the 75th percentile of the gap field over the surface support
    # (cells with at least 5% of the peak surface density). When evals densely cover
    # surfaces, gap is near zero everywhere on the support and that quantile collapses
    # to 0, so we fall back to half the peak gap. The semantics flip silently between
    # the two regimes — this is intentional but means a coverage of 1.0 and a coverage
    # of 0.5 use slightly different threshold definitions.
    support_mask = rho_S > rho_S.max() * 0.05
    if support_mask.any():
        threshold = float(np.quantile(gap[support_mask], threshold_quantile))
    else:
        threshold = float(np.quantile(gap, threshold_quantile))
    if threshold <= 0:
        threshold = float(gap.max() * 0.5) if gap.max() > 0 else 0.0

    regions = _find_regions(gap, threshold, grid_x, grid_y)
    surface_mass = float(_integrate(rho_S, grid_x, grid_y))
    gap_mass = float(_integrate(gap, grid_x, grid_y))
    coverage = 1.0 - (gap_mass / surface_mass) if surface_mass > 0 else 0.0
    coverage = max(0.0, min(1.0, coverage))

    return GapField(
        grid_x=grid_x,
        grid_y=grid_y,
        rho_surfaces=rho_S,
        rho_evals=rho_E,
        gap_field=gap,
        threshold=threshold,
        alpha=float(alpha),
        bandwidth=float(bandwidth),
        regions=regions,
        coverage_score=float(coverage),
    )


def _scott_bandwidth(points: NDArray[np.float64]) -> float:
    n = max(1, len(points))
    std = float(points.std(axis=0).mean())
    return max(std * (n ** (-1.0 / 6.0)), 1e-2)


def _gaussian_kde(
    points: NDArray[np.float64],
    grid_points: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    if points.size == 0:
        return np.zeros(len(grid_points))
    diff = grid_points[:, None, :] - points[None, :, :]
    sq = (diff ** 2).sum(axis=2)
    weights = np.exp(-sq / (2.0 * bandwidth * bandwidth))
    norm = (2.0 * np.pi * bandwidth * bandwidth)
    return (weights.sum(axis=1)) / (norm * len(points))


def _normalize_density(
    field: NDArray[np.float64], grid_x: NDArray, grid_y: NDArray
) -> NDArray[np.float64]:
    integral = _integrate(field, grid_x, grid_y)
    if integral <= 0:
        return field
    return field / integral


def _integrate(field: NDArray[np.float64], grid_x: NDArray, grid_y: NDArray) -> float:
    dx = float(grid_x[1] - grid_x[0]) if len(grid_x) > 1 else 1.0
    dy = float(grid_y[1] - grid_y[0]) if len(grid_y) > 1 else 1.0
    return float(field.sum() * dx * dy)


def _find_regions(
    gap: NDArray[np.float64],
    threshold: float,
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
) -> list[GapRegion]:
    mask = gap > threshold
    if not mask.any():
        return []
    labels = _connected_components(mask)
    n_labels = int(labels.max())
    dx = float(grid_x[1] - grid_x[0]) if len(grid_x) > 1 else 1.0
    dy = float(grid_y[1] - grid_y[0]) if len(grid_y) > 1 else 1.0
    cell_area = dx * dy
    out: list[GapRegion] = []
    for label_id in range(1, n_labels + 1):
        component_mask = labels == label_id
        if not component_mask.any():
            continue
        rows, cols = np.where(component_mask)
        ys = grid_y[rows]
        xs = grid_x[cols]
        weights = gap[component_mask]
        total = float(weights.sum())
        if total <= 0:
            continue
        cx = float((xs * weights).sum() / total)
        cy = float((ys * weights).sum() / total)
        out.append(
            GapRegion(
                id=f"R{label_id}",
                centroid_2d=(cx, cy),
                bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                integrated_mass=total * cell_area,
                area_cells=int(component_mask.sum()),
            )
        )
    out.sort(key=lambda r: r.integrated_mass, reverse=True)
    for rank, region in enumerate(out, start=1):
        region.id = f"R{rank}"
    return out


def _connected_components(mask: NDArray[np.bool_]) -> NDArray[np.int_]:
    try:
        from scipy.ndimage import label  # type: ignore[import-not-found]

        labels, _ = label(mask)
        return labels.astype(int)
    except Exception:
        return _flood_fill(mask)


def _flood_fill(mask: NDArray[np.bool_]) -> NDArray[np.int_]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    next_label = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labels[i, j] == 0:
                next_label += 1
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if 0 <= y < h and 0 <= x < w and mask[y, x] and labels[y, x] == 0:
                        labels[y, x] = next_label
                        stack.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
    return labels


def nearest_surface_indices(
    gap_centroid_2d: tuple[float, float],
    surface_points_2d: NDArray[np.float64],
    surface_embeddings_high: NDArray[np.float64],
    k: int = 5,
) -> list[int]:
    """k nearest surfaces to a gap centroid. Distance computed in the 2D projection."""
    if surface_points_2d.size == 0:
        return []
    cx, cy = gap_centroid_2d
    diffs = surface_points_2d - np.asarray([cx, cy])
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    order = np.argsort(dists)[: max(1, k)]
    return [int(i) for i in order]

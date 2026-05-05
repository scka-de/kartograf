from __future__ import annotations

from .models import Region


def compute_coverage(regions: list[Region]) -> float:
    denominator = sum(max(0.0, r.density) for r in regions)
    if denominator <= 0:
        return 0.0
    numerator = sum(r.density for r in regions if r.eval_case_ids)
    return max(0.0, min(1.0, numerator / denominator))


def compute_noise_fraction(total_count: int, noise_count: int) -> float:
    if total_count <= 0:
        return 0.0
    return max(0.0, min(1.0, noise_count / total_count))


def coverage_warnings(noise_fraction: float, regions: list[Region]) -> list[str]:
    warnings: list[str] = []
    if noise_fraction > 0.20:
        warnings.append("noise_fraction is high - coverage may be less reliable")
    if any(region.member_count < 5 for region in regions):
        warnings.append("very small regions detected")
    if len(regions) < 3:
        warnings.append("corpus produced too few regions")
    return warnings

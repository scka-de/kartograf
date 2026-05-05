from cartograph.core.coverage import compute_coverage, compute_noise_fraction, coverage_warnings


def test_compute_coverage_weights_by_region_density(tiny_regions):
    assert compute_coverage(tiny_regions) == 0.6


def test_noise_fraction_and_warnings(tiny_regions):
    assert compute_noise_fraction(10, 3) == 0.3
    warnings = coverage_warnings(0.3, tiny_regions)
    assert "noise_fraction is high" in warnings[0]


def test_empty_region_coverage_is_zero():
    assert compute_coverage([]) == 0.0


def test_noise_fraction_handles_empty_total():
    assert compute_noise_fraction(0, 10) == 0.0

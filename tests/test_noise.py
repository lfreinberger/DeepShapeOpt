"""ECnoise port: recovers a known noise level and flags unusable step sizes."""

import numpy as np

from deepshapeopt.diagnostics.noise import H_TOO_LARGE, H_TOO_SMALL, NOISE_DETECTED, ecnoise, ecnoise_vector

T = np.arange(9) - 4.0  # offsets t_i of 9 equally spaced points
H = 0.01


def _smooth(t):
    return 1.0 + 0.3 * t + 0.05 * t ** 2 - 0.01 * t ** 3


def test_scalar_recovers_sigma():
    sigma = 1e-4
    estimates, informs = [], []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        f = _smooth(H * T) + sigma * rng.standard_normal(T.size)
        noise, _, inform = ecnoise(f)
        informs.append(inform)
        if inform == NOISE_DETECTED:
            estimates.append(noise)
    assert np.mean(np.asarray(informs) == NOISE_DETECTED) > 0.8
    med = np.median(estimates)
    assert 0.5 * sigma < med < 2.0 * sigma, med


def test_scalar_flags_h_too_small_and_too_large():
    _, _, inform = ecnoise(np.full(9, 3.0))
    assert inform == H_TOO_SMALL
    _, _, inform = ecnoise(_smooth(1.0 * T))  # 20%+ relative range, no noise
    assert inform == H_TOO_LARGE


def test_vector_recovers_noise_norm():
    sigma, n_vars = 1e-3, 500
    rng = np.random.default_rng(0)
    slopes = rng.standard_normal(n_vars)
    G = 0.2 + np.outer(H * T, slopes) + sigma * rng.standard_normal((T.size, n_vars))
    noise, _, inform = ecnoise_vector(G)
    assert inform == NOISE_DETECTED
    expected = sigma * np.sqrt(n_vars)
    assert 0.5 * expected < noise < 2.0 * expected, (noise, expected)


def test_vector_flags_h_too_small():
    _, _, inform = ecnoise_vector(np.ones((9, 50)))
    assert inform == H_TOO_SMALL

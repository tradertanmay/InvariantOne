"""
Unit tests for calibration metrics and post-hoc temperature scaling.
"""

from invariantone.metrics.calibration import TemperatureScaler, expected_calibration_error


def test_temperature_scaler_fitting():
    scaler = TemperatureScaler()

    # Synthetic predictions: model is overconfident
    # True label is class 0 half the time, class 1 half the time
    raw_scores = [
        {"A": 5.0, "B": -5.0},
        {"A": 5.0, "B": -5.0},
        {"A": -5.0, "B": 5.0},
        {"A": -5.0, "B": 5.0},
    ]
    targets = ["A", "B", "A", "B"]

    # Initial temperature is 1.0
    assert scaler.temperature == 1.0

    # Fit on calibration split
    fitted_tau = scaler.fit(raw_scores, targets)
    assert fitted_tau > 1.0  # Temperature should increase to soften overconfident probabilities

    scaled_probs = scaler.scale_probabilities(raw_scores[0])
    assert scaled_probs["A"] < 0.99  # Softened from ~0.99995
    assert abs(sum(scaled_probs.values()) - 1.0) < 1e-4


def test_expected_calibration_error():
    # Perfectly calibrated predictions:
    # 10 cases with confidence 0.8, exactly 8 correct
    probs = [{"A": 0.8, "B": 0.2} for _ in range(10)]
    targets = ["A"] * 8 + ["B"] * 2

    ece = expected_calibration_error(probs, targets, num_bins=10)
    assert ece < 0.05

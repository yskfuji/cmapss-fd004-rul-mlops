import numpy as np
from models.gbdt_pipeline import rolling_features, rul_sample_weights


def test_rul_sample_weights_normalized_mean():
    weights = rul_sample_weights(np.asarray([0.0, 10.0, 125.0]))
    assert np.isclose(weights.mean(), 1.0)


def test_rolling_features_shape_is_consistent():
    window = np.ones((30, 2), dtype=float)
    features = rolling_features(window)
    assert features.shape == (54,)

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.anomaly_detection import (
    DenseAutoencoderDetector,
    score_anomalies,
    train_dense_autoencoder,
)


@pytest.fixture
def training_features():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(128, 3)) * [1, 1000, 5] + [5, 20000, -10],
        columns=["a", "b", "c"],
    )


@pytest.fixture
def evaluation_features():
    return pd.DataFrame(
        {"a": [5.0, 100.0], "b": [20000.0, 100000.0], "c": [-10.0, 1000.0]},
        index=pd.Index([9, 9], name="window"),
    )


@pytest.fixture
def model(training_features):
    return train_dense_autoencoder(training_features)


def test_exact_architecture_and_configuration(model):
    assert isinstance(model, DenseAutoencoderDetector)
    assert isinstance(model.scaler_, StandardScaler)
    assert isinstance(model.network_, MLPRegressor)
    params = model.network_.get_params()
    expected = {
        "hidden_layer_sizes": (32, 16, 8, 16, 32),
        "activation": "relu", "solver": "adam", "loss": "squared_error",
        "learning_rate_init": 0.001, "batch_size": 128, "max_iter": 100,
        "alpha": 0.0001, "random_state": 42,
        "early_stopping": False, "shuffle": True,
    }
    for key, value in expected.items():
        assert params[key] == value
    assert [weights.shape for weights in model.network_.coefs_] == [
        (3, 32), (32, 16), (16, 8), (8, 16), (16, 32), (32, 3),
    ]


def test_fit_uses_scaled_train_as_input_and_target(training_features, monkeypatch):
    original = training_features.copy(deep=True)
    expected = StandardScaler().fit_transform(training_features)
    calls = []
    original_fit = MLPRegressor.fit

    def fit(network, X, y):
        np.testing.assert_array_equal(X, expected)
        np.testing.assert_array_equal(y, expected)
        calls.append(len(X))
        return original_fit(network, X, y)

    monkeypatch.setattr(MLPRegressor, "fit", fit)
    detector = DenseAutoencoderDetector()
    assert detector.fit(training_features) is detector
    assert calls == [len(training_features)]
    pd.testing.assert_frame_equal(training_features, original)


def test_scaler_and_threshold_learned_only_from_train(model, training_features):
    assert model.scaler_.n_samples_seen_ == len(training_features)
    np.testing.assert_allclose(model.scaler_.mean_, training_features.mean())
    np.testing.assert_allclose(model.scaler_.var_, training_features.var(ddof=0))
    scaled = model.scaler_.transform(training_features)
    errors = np.mean((scaled - model.network_.predict(scaled)) ** 2, axis=1)
    assert model.threshold_ == np.percentile(errors, 95)


def test_exact_mse_and_score_anomalies_contract(model, evaluation_features):
    scaled = model.scaler_.transform(evaluation_features)
    errors = np.mean((scaled - model.network_.predict(scaled)) ** 2, axis=1)
    np.testing.assert_array_equal(model.score_samples(evaluation_features), -errors)
    result = score_anomalies(model, evaluation_features)
    np.testing.assert_array_equal(result.anomaly_score, errors)
    np.testing.assert_array_equal(result.is_anomaly, errors > model.threshold_)
    assert result.columns.tolist() == ["anomaly_score", "is_anomaly"]
    assert result.is_anomaly.dtype == bool
    pd.testing.assert_index_equal(result.index, evaluation_features.index)
    assert errors[1] > errors[0]


def test_predict_uses_strict_threshold(model, evaluation_features, monkeypatch):
    monkeypatch.setattr(
        model, "score_samples",
        lambda features: -np.array([model.threshold_, np.nextafter(model.threshold_, np.inf)]),
    )
    np.testing.assert_array_equal(model.predict(evaluation_features), [1, -1])


def test_scoring_never_fits_or_changes_state(model, evaluation_features, monkeypatch):
    before = pickle.dumps(model)
    original = evaluation_features.copy(deep=True)

    def forbid_fit(*args, **kwargs):
        pytest.fail("Evaluación no debe ajustar scaler ni red")

    with monkeypatch.context() as patch:
        for cls in (DenseAutoencoderDetector, StandardScaler, MLPRegressor):
            for method in ("fit", "fit_transform", "partial_fit"):
                patch.setattr(cls, method, forbid_fit, raising=False)
        first = score_anomalies(model, evaluation_features)
        score_anomalies(model, evaluation_features + 1e6)
        second = score_anomalies(model, evaluation_features)
        pd.testing.assert_frame_equal(first, second)

    assert pickle.dumps(model) == before
    pd.testing.assert_frame_equal(evaluation_features, original)


def test_reproducibility(model, training_features, evaluation_features):
    second = train_dense_autoencoder(training_features)
    np.testing.assert_array_equal(
        model.score_samples(evaluation_features), second.score_samples(evaluation_features),
    )
    assert model.threshold_ == second.threshold_
    for first_weights, second_weights in zip(model.network_.coefs_, second.network_.coefs_):
        np.testing.assert_array_equal(first_weights, second_weights)


@pytest.mark.parametrize(
    "features, error",
    [
        (pd.DataFrame(), ValueError),
        (pd.DataFrame(columns=["a", "b", "c"]), ValueError),
        (pd.DataFrame({"a": [np.nan]}), ValueError),
        (pd.DataFrame({"a": [np.inf]}), ValueError),
        (pd.DataFrame({"a": [-np.inf]}), ValueError),
        (pd.DataFrame({"a": ["text"]}), TypeError),
    ],
)
def test_invalid_inputs_in_fit_and_scoring(model, features, error):
    with pytest.raises(error):
        train_dense_autoencoder(features)
    for method in (model.score_samples, model.predict):
        with pytest.raises(error):
            method(features)


@pytest.mark.parametrize("columns", [
    ["a", "b"], ["b", "a", "c"], ["a", "b", "d"], ["a", "b", "c", "d"],
])
def test_rejects_incompatible_schema(model, columns):
    evaluation = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
    for method in (model.score_samples, model.predict):
        with pytest.raises(ValueError, match="features"):
            method(evaluation)


def test_unfitted_detector_rejected(evaluation_features):
    for method in ("score_samples", "predict"):
        with pytest.raises(NotFittedError):
            getattr(DenseAutoencoderDetector(), method)(evaluation_features)


def test_single_feature_reconstruction_shape(training_features):
    train = training_features[["a"]]
    model = train_dense_autoencoder(train)
    evaluation = pd.DataFrame({"a": [5.0, 100.0]})
    scaled = model.scaler_.transform(evaluation)
    reconstruction = model.network_.predict(scaled).reshape(-1, 1)
    expected = np.mean((scaled - reconstruction) ** 2, axis=1)
    np.testing.assert_array_equal(-model.score_samples(evaluation), expected)

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from predictive_maintenance.anomaly_detection import (
    score_anomalies,
    train_one_class_svm,
)


@pytest.fixture
def training_features():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(100, 2)) * [1.0, 1000.0] + [5.0, 20000.0],
        columns=["a", "b"],
    )


def test_train_one_class_svm_fits_without_labels(training_features):
    model = train_one_class_svm(training_features, nu=0.1, gamma=0.25)

    assert isinstance(model, Pipeline)
    scaler = model.named_steps["scaler"]
    detector = model.named_steps["ocsvm"]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(detector, OneClassSVM)
    assert model.n_features_in_ == 2
    assert detector.support_vectors_.shape[1] == 2
    assert detector.nu == 0.1
    assert detector.gamma == 0.25
    np.testing.assert_allclose(scaler.mean_, training_features.mean())
    np.testing.assert_allclose(scaler.var_, training_features.var(ddof=0))
    np.testing.assert_allclose(
        scaler.transform(training_features).std(axis=0), [1.0, 1.0]
    )


def test_scoring_preserves_contract_and_anomaly_direction(training_features):
    model = train_one_class_svm(training_features, nu=0.1)
    evaluation = pd.DataFrame(
        {"a": [5.0, 5.2, 100.0], "b": [20000.0, 20100.0, 100000.0]},
        index=pd.Index([21, 8, 21], name="window"),
    )
    result = score_anomalies(model, evaluation)

    assert result.columns.tolist() == ["anomaly_score", "is_anomaly"]
    pd.testing.assert_index_equal(result.index, evaluation.index)
    assert result["is_anomaly"].dtype == bool
    assert np.isfinite(result["anomaly_score"]).all()
    assert result["anomaly_score"].nunique() == 3
    assert result.iloc[-1]["anomaly_score"] > result.iloc[0]["anomaly_score"]
    assert result.iloc[-1]["is_anomaly"]
    scaled = model.named_steps["scaler"].transform(evaluation)
    detector = model.named_steps["ocsvm"]
    np.testing.assert_allclose(result["anomaly_score"], -detector.score_samples(scaled))
    np.testing.assert_array_equal(result["is_anomaly"], detector.predict(scaled) == -1)


def test_scaler_is_fitted_only_on_train_and_never_refitted(
    training_features, monkeypatch,
):
    model = train_one_class_svm(training_features)
    scaler = model.named_steps["scaler"]
    mean, variance, scale = scaler.mean_.copy(), scaler.var_.copy(), scaler.scale_.copy()
    assert scaler.n_samples_seen_ == len(training_features)

    def forbid_fit(*args, **kwargs):
        pytest.fail("El scoring no debe ajustar el scaler")

    monkeypatch.setattr(scaler, "fit", forbid_fit)
    monkeypatch.setattr(scaler, "fit_transform", forbid_fit)
    monkeypatch.setattr(scaler, "partial_fit", forbid_fit)
    evaluation = training_features.iloc[:3].copy()
    original = evaluation.copy(deep=True)
    first = score_anomalies(model, evaluation)
    score_anomalies(model, evaluation + 1e9)
    second = score_anomalies(model, evaluation)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(evaluation, original)
    np.testing.assert_array_equal(scaler.mean_, mean)
    np.testing.assert_array_equal(scaler.var_, variance)
    np.testing.assert_array_equal(scaler.scale_, scale)
    assert scaler.n_samples_seen_ == len(training_features)


@pytest.mark.parametrize(
    "features, error",
    [
        (pd.DataFrame(), ValueError),
        (pd.DataFrame(columns=["a", "b"]), ValueError),
        (pd.DataFrame({"a": [np.nan], "b": [1.0]}), ValueError),
        (pd.DataFrame({"a": [np.inf], "b": [1.0]}), ValueError),
        (pd.DataFrame({"a": [-np.inf], "b": [1.0]}), ValueError),
        (pd.DataFrame({"a": ["text"], "b": [1.0]}), TypeError),
    ],
)
def test_invalid_features_rejected_in_training_and_scoring(
    training_features, features, error,
):
    with pytest.raises(error):
        train_one_class_svm(features)
    model = train_one_class_svm(training_features)
    with pytest.raises(error):
        score_anomalies(model, features)


@pytest.mark.parametrize("nu", [0.0, -0.1, 1.1, np.nan])
def test_invalid_nu_rejected(training_features, nu):
    with pytest.raises(ValueError, match="nu"):
        train_one_class_svm(training_features, nu=nu)


@pytest.mark.parametrize("gamma", [np.nan, -1.0, "invalid"])
def test_invalid_gamma_rejected(training_features, gamma):
    with pytest.raises(ValueError, match="gamma"):
        train_one_class_svm(training_features, gamma=gamma)


@pytest.mark.parametrize("columns", [["a"], ["b", "a"], ["a", "c"]])
def test_scoring_rejects_changed_feature_schema(training_features, columns):
    model = train_one_class_svm(training_features)
    evaluation = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
    with pytest.raises(ValueError, match="feature"):
        score_anomalies(model, evaluation)

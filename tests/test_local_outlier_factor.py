import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.anomaly_detection import (
    score_anomalies,
    train_local_outlier_factor,
)


@pytest.fixture
def training_features():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(100, 2)) * [1.0, 1000.0] + [5.0, 20000.0],
        columns=["a", "b"],
    )


@pytest.fixture
def evaluation_features():
    return pd.DataFrame(
        {"a": [5.0, 5.2, 100.0], "b": [20000.0, 20100.0, 100000.0]},
        index=pd.Index([21, 8, 21], name="window"),
    )


def test_training_without_labels_fits_baseline_pipeline(training_features):
    original = training_features.copy(deep=True)
    model = train_local_outlier_factor(training_features)

    assert isinstance(model, Pipeline)
    scaler = model.named_steps["scaler"]
    detector = model.named_steps["lof"]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(detector, LocalOutlierFactor)
    assert detector.novelty is True
    assert detector.n_neighbors == 20
    assert detector.contamination == "auto"
    assert detector.get_params() == LocalOutlierFactor(novelty=True).get_params()
    assert model.n_features_in_ == 2
    assert detector.n_neighbors_ == 20
    assert len(detector.negative_outlier_factor_) == len(training_features)
    np.testing.assert_allclose(scaler.mean_, training_features.mean())
    np.testing.assert_allclose(scaler.var_, training_features.var(ddof=0))
    np.testing.assert_allclose(
        scaler.transform(training_features).std(axis=0), [1.0, 1.0]
    )
    pd.testing.assert_frame_equal(training_features, original)


def test_scoring_new_data_preserves_contract_and_direction(
    training_features, evaluation_features,
):
    model = train_local_outlier_factor(training_features)
    result = score_anomalies(model, evaluation_features)

    assert len(result) == len(evaluation_features)
    assert result.columns.tolist() == ["anomaly_score", "is_anomaly"]
    pd.testing.assert_index_equal(result.index, evaluation_features.index)
    assert result["is_anomaly"].dtype == bool
    assert np.isfinite(result["anomaly_score"]).all()
    assert result["anomaly_score"].nunique() == 3
    assert result.iloc[-1]["anomaly_score"] > result.iloc[:-1]["anomaly_score"].max()
    assert result.iloc[-1]["is_anomaly"]
    assert not result.iloc[0]["is_anomaly"]
    scaled = model.named_steps["scaler"].transform(evaluation_features)
    detector = model.named_steps["lof"]
    np.testing.assert_allclose(result["anomaly_score"], -detector.score_samples(scaled))
    np.testing.assert_array_equal(result["is_anomaly"], detector.predict(scaled) == -1)


def test_evaluation_never_refits_scaler(
    training_features, evaluation_features, monkeypatch,
):
    model = train_local_outlier_factor(training_features)
    scaler = model.named_steps["scaler"]
    mean, variance, scale = scaler.mean_.copy(), scaler.var_.copy(), scaler.scale_.copy()
    assert scaler.n_samples_seen_ == len(training_features)

    def forbid_fit(*args, **kwargs):
        pytest.fail("El scoring no debe ajustar el scaler")

    monkeypatch.setattr(scaler, "fit", forbid_fit)
    monkeypatch.setattr(scaler, "fit_transform", forbid_fit)
    monkeypatch.setattr(scaler, "partial_fit", forbid_fit)
    original = evaluation_features.copy(deep=True)
    first = score_anomalies(model, evaluation_features)
    score_anomalies(model, evaluation_features + 1e9)
    second = score_anomalies(model, evaluation_features)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(evaluation_features, original)
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
        train_local_outlier_factor(features)
    model = train_local_outlier_factor(training_features)
    with pytest.raises(error):
        score_anomalies(model, features)


@pytest.mark.parametrize(
    "columns", [["a"], ["b", "a"], ["a", "c"], ["a", "b", "c"]],
)
def test_scoring_rejects_incompatible_schema(training_features, columns):
    model = train_local_outlier_factor(training_features)
    evaluation = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
    with pytest.raises(ValueError, match="feature"):
        score_anomalies(model, evaluation)

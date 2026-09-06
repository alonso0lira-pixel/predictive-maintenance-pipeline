import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import predictive_maintenance.model_comparison as comparison
from predictive_maintenance.anomaly_detection import DenseAutoencoderDetector
from predictive_maintenance.experiment import run_anomaly_experiment
from predictive_maintenance.ground_truth import get_failure_intervals
from predictive_maintenance.labeling import label_failure_windows
from predictive_maintenance.modeling import MODEL_INPUT_COLUMNS, prepare_model_input


MODELS = [
    "isolation_forest", "one_class_svm", "local_outlier_factor", "dense_autoencoder",
]
GLOBAL_COLUMNS = [
    "model", "rows", "positives", "negatives", "prevalence", "roc_auc",
    "average_precision", "training_seconds", "scoring_seconds",
]
FAILURE_COLUMNS = [
    "model", "failure_id", "failure_windows", "roc_auc", "average_precision",
    "score_mean", "score_median", "score_max",
]


@pytest.fixture
def datasets(tmp_path):
    rng = np.random.default_rng(42)

    def frame(starts):
        data = pd.DataFrame(
            rng.normal(size=(len(starts), len(MODEL_INPUT_COLUMNS))),
            columns=MODEL_INPUT_COLUMNS,
        )
        data["segment_id"] = 0
        data["start_index"] = np.arange(len(starts)) * 60
        data["end_index"] = data["start_index"] + 60
        data["window_start_timestamp"] = starts
        data["window_end_timestamp"] = starts + pd.Timedelta(minutes=10)
        # Etiquetas preexistentes no deben entrar en el entrenamiento.
        data["is_failure"] = True
        data.index = np.arange(len(data))[::-1] * 3
        return data

    train = frame(pd.date_range("2020-02-01", periods=40, freq="10min"))
    starts = []
    for failure in get_failure_intervals().itertuples():
        for hours in (48, 18, 9, 4, 2, 0.5):
            starts.append(failure.start_timestamp - pd.Timedelta(hours=hours))
        # Exactamente 50 % de solapamiento, además de una ventana interior.
        starts.extend([
            failure.start_timestamp - pd.Timedelta(minutes=5),
            failure.start_timestamp + pd.Timedelta(minutes=10),
        ])
    evaluation = frame(pd.DatetimeIndex(starts))
    train_path = tmp_path / "train.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    train.to_parquet(train_path)
    evaluation.to_parquet(evaluation_path)
    return train_path, evaluation_path, train, evaluation


def test_comparison_contract_and_original_experiment_equivalence(datasets):
    train_path, evaluation_path, _, evaluation = datasets
    report = comparison.run_model_comparison(train_path, evaluation_path)
    table = report["model_comparison"]
    failures = report["model_failure_metrics"]
    assert table.columns.tolist() == GLOBAL_COLUMNS
    assert table.model.tolist() == MODELS
    assert table.rows.tolist() == [len(evaluation)] * 4
    assert table.positives.tolist() == [8] * 4
    assert table.negatives.tolist() == [24] * 4
    assert table.prevalence.tolist() == [0.25] * 4
    assert (table[["training_seconds", "scoring_seconds"]] >= 0).all().all()
    assert failures.columns.tolist() == FAILURE_COLUMNS
    assert len(failures) == 16
    for name in MODELS:
        subset = failures[failures.model.eq(name)]
        assert subset.failure_id.tolist() == [1, 2, 3, 4]
        assert subset.failure_windows.tolist() == [2] * 4
    local = report["model_local_horizon_metrics"]
    assert len(local) == 96
    assert local.model.unique().tolist() == MODELS
    for name in MODELS:
        subset = local[local.model.eq(name)]
        assert subset.groupby("failure_id").size().to_dict() == {
            1: 6, 2: 6, 3: 6, 4: 6,
        }
    assert "output_files" not in report

    original = run_anomaly_experiment(train_path, evaluation_path)
    assert set(original) == {"global_metrics", "failure_metrics", "local_horizon_metrics"}
    for key, value in original["global_metrics"].items():
        assert table.iloc[0][key] == value
    pd.testing.assert_frame_equal(
        failures[failures.model.eq("isolation_forest")].drop(columns="model").reset_index(drop=True),
        original["failure_metrics"],
    )
    local = report["model_local_horizon_metrics"]
    pd.testing.assert_frame_equal(
        local[local.model.eq("isolation_forest")].drop(columns="model").reset_index(drop=True),
        original["local_horizon_metrics"],
    )


def test_identical_inputs_labels_and_train_only_fit(datasets, monkeypatch):
    train_path, evaluation_path, train, evaluation = datasets
    X_train, _ = prepare_model_input(train)
    X_evaluation, metadata = prepare_model_input(evaluation)
    expected_labels = label_failure_windows(metadata, overlap_threshold=0.50).reset_index(drop=True)
    fit_calls = []

    def guard_fit(cls):
        original_fit = cls.fit

        def fit(self, X, y=None, **kwargs):
            if cls is MLPRegressor:
                np.testing.assert_array_equal(y, X)
            else:
                assert y is None
            if cls in (IsolationForest, StandardScaler):
                pd.testing.assert_frame_equal(X, X_train)
            else:
                expected = (X_train - X_train.mean()) / X_train.std(ddof=0)
                np.testing.assert_allclose(X, expected, atol=1e-12)
            if cls is IsolationForest:
                assert self.n_estimators == 100 and self.random_state == 42
            elif cls is OneClassSVM:
                assert self.nu == 0.05 and self.gamma == "scale"
            elif cls is LocalOutlierFactor:
                assert self.n_neighbors == 20
                assert self.contamination == "auto" and self.novelty is True
            fit_calls.append(cls)
            return original_fit(self, X, y=y, **kwargs)

        monkeypatch.setattr(cls, "fit", fit)

    for cls in (IsolationForest, StandardScaler, OneClassSVM, LocalOutlierFactor, MLPRegressor):
        guard_fit(cls)
    original_autoencoder_fit = DenseAutoencoderDetector.fit
    autoencoder_inputs = []

    def autoencoder_fit(self, features):
        pd.testing.assert_frame_equal(features, X_train)
        autoencoder_inputs.append(features)
        return original_autoencoder_fit(self, features)

    monkeypatch.setattr(DenseAutoencoderDetector, "fit", autoencoder_fit)
    original_score = comparison.score_anomalies
    scored_inputs = []

    def score(model, features):
        pd.testing.assert_frame_equal(features, X_evaluation)
        scored_inputs.append(features)
        return original_score(model, features)

    monkeypatch.setattr(comparison, "score_anomalies", score)
    evaluated = []
    original_evaluate = comparison.evaluate_global_scores

    def evaluate(results):
        pd.testing.assert_frame_equal(results[expected_labels.columns], expected_labels)
        evaluated.append(results)
        return original_evaluate(results)

    monkeypatch.setattr(comparison, "evaluate_global_scores", evaluate)
    comparison.run_model_comparison(train_path, evaluation_path)
    assert len(evaluated) == len(scored_inputs) == 4
    assert all(features is scored_inputs[0] for features in scored_inputs)
    assert fit_calls.count(StandardScaler) == 3
    assert len(autoencoder_inputs) == 1
    for cls in (IsolationForest, OneClassSVM, LocalOutlierFactor, MLPRegressor):
        assert fit_calls.count(cls) == 1


def test_save_comparison_csv_contents(datasets, tmp_path):
    train_path, evaluation_path, _, _ = datasets
    output_dir = tmp_path / "results"
    report = comparison.run_model_comparison(train_path, evaluation_path, output_dir)
    assert set(report["output_files"]) == {"model_comparison", "model_failure_metrics"}
    assert {p.name for p in output_dir.iterdir()} == {
        "model_comparison.csv", "model_failure_metrics.csv",
    }
    for key, path in report["output_files"].items():
        saved = pd.read_csv(path)
        pd.testing.assert_frame_equal(saved, report[key])
        assert saved.model.unique().tolist() == MODELS


def test_comparison_rejects_mismatched_scores(datasets, monkeypatch):
    train_path, evaluation_path, _, _ = datasets
    original_score = comparison.score_anomalies
    monkeypatch.setattr(
        comparison, "score_anomalies",
        lambda model, features: original_score(model, features).iloc[:-1],
    )
    with pytest.raises(ValueError, match="número de scores"):
        comparison.run_model_comparison(train_path, evaluation_path)

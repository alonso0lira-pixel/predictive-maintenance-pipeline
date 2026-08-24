import pandas as pd
import pytest

import predictive_maintenance.experiment as experiment_module


def test_run_anomaly_experiment_orchestrates_pipeline(
    monkeypatch,
) -> None:
    train = pd.DataFrame({"dataset": ["train"]})
    evaluation = pd.DataFrame(
        {
            "dataset": [
                "evaluation",
                "evaluation",
            ]
        }
    )

    def fake_read_parquet(path):
        if "train" in str(path):
            return train

        return evaluation

    X_train = pd.DataFrame(
        {
            "feature": [
                0.1,
                0.2,
            ]
        }
    )

    X_evaluation = pd.DataFrame(
        {
            "feature": [
                0.3,
                0.9,
            ]
        }
    )

    metadata = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-17"),
                pd.Timestamp("2020-04-18"),
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-04-17 00:10"),
                pd.Timestamp("2020-04-18 00:10"),
            ],
        }
    )

    def fake_prepare_model_input(df):
        if df is train:
            return X_train, pd.DataFrame()

        return X_evaluation, metadata

    model = object()

    def fake_train(
        features,
        n_estimators,
        random_state,
    ):
        assert features is X_train
        assert n_estimators == 200
        assert random_state == 7

        return model

    def fake_score(model_arg, features):
        assert model_arg is model
        assert features is X_evaluation

        return pd.DataFrame(
            {
                "anomaly_score": [
                    0.2,
                    0.9,
                ],
                "is_anomaly": [
                    False,
                    True,
                ],
            }
        )

    def fake_label(
        windows,
        overlap_threshold,
    ):
        assert windows is metadata
        assert overlap_threshold == pytest.approx(0.60)

        labeled = windows.copy()
        labeled["failure_overlap_ratio"] = [
            0.0,
            1.0,
        ]
        labeled["failure_id"] = pd.Series(
            [
                pd.NA,
                1,
            ],
            dtype="Int64",
        )
        labeled["is_failure"] = [
            False,
            True,
        ]

        return labeled

    global_metrics = {
        "roc_auc": 1.0,
        "pr_auc": 1.0,
    }

    failure_metrics = pd.DataFrame(
        {
            "failure_id": [1],
            "roc_auc": [1.0],
        }
    )

    local_metrics = pd.DataFrame(
        {
            "failure_id": [1],
            "period": ["durante"],
            "roc_auc": [1.0],
        }
    )

    monkeypatch.setattr(
        experiment_module.pd,
        "read_parquet",
        fake_read_parquet,
    )
    monkeypatch.setattr(
        experiment_module,
        "prepare_model_input",
        fake_prepare_model_input,
    )
    monkeypatch.setattr(
        experiment_module,
        "train_isolation_forest",
        fake_train,
    )
    monkeypatch.setattr(
        experiment_module,
        "score_anomalies",
        fake_score,
    )
    monkeypatch.setattr(
        experiment_module,
        "label_failure_windows",
        fake_label,
    )
    monkeypatch.setattr(
        experiment_module,
        "get_failure_intervals",
        lambda: pd.DataFrame(
            {
                "failure_id": [1],
                "start_timestamp": [
                    pd.Timestamp("2020-04-18")
                ],
                "end_timestamp": [
                    pd.Timestamp("2020-04-18 01:00")
                ],
            }
        ),
    )
    monkeypatch.setattr(
        experiment_module,
        "evaluate_global_scores",
        lambda results: global_metrics,
    )
    monkeypatch.setattr(
        experiment_module,
        "evaluate_scores_by_failure",
        lambda results: failure_metrics,
    )
    monkeypatch.setattr(
        experiment_module,
        "evaluate_local_horizons",
        lambda results, failures: local_metrics,
    )

    report = experiment_module.run_anomaly_experiment(
        "train.parquet",
        "evaluation.parquet",
        n_estimators=200,
        random_state=7,
        overlap_threshold=0.60,
    )

    assert report["global_metrics"] is global_metrics
    assert report["failure_metrics"] is failure_metrics
    assert (
        report["local_horizon_metrics"]
        is local_metrics
    )


def test_run_anomaly_experiment_rejects_mismatched_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        experiment_module.pd,
        "read_parquet",
        lambda path: pd.DataFrame({"x": [1]}),
    )

    monkeypatch.setattr(
        experiment_module,
        "prepare_model_input",
        lambda df: (
            pd.DataFrame({"feature": [0.1]}),
            pd.DataFrame(
                {
                    "window_start_timestamp": [
                        pd.Timestamp("2020-01-01")
                    ],
                    "window_end_timestamp": [
                        pd.Timestamp("2020-01-01 00:10")
                    ],
                }
            ),
        ),
    )

    monkeypatch.setattr(
        experiment_module,
        "train_isolation_forest",
        lambda *args, **kwargs: object(),
    )

    monkeypatch.setattr(
        experiment_module,
        "score_anomalies",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "anomaly_score": [
                    0.1,
                    0.2,
                ]
            }
        ),
    )

    monkeypatch.setattr(
        experiment_module,
        "label_failure_windows",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "is_failure": [False],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="número de scores",
    ):
        experiment_module.run_anomaly_experiment(
            "train.parquet",
            "evaluation.parquet",
        )

def test_save_experiment_results_creates_files(
    tmp_path,
) -> None:
    report = {
        "global_metrics": {
            "rows": 100,
            "positives": 10,
            "roc_auc": 0.95,
            "pr_auc": 0.40,
        },
        "failure_metrics": pd.DataFrame(
            {
                "failure_id": [1, 2],
                "roc_auc": [0.90, 0.95],
            }
        ),
        "local_horizon_metrics": pd.DataFrame(
            {
                "failure_id": [1],
                "period": ["1-0h"],
                "roc_auc": [0.80],
            }
        ),
    }

    paths = experiment_module.save_experiment_results(
        report,
        tmp_path,
    )

    assert paths["global_metrics"].exists()
    assert paths["failure_metrics"].exists()
    assert paths["local_horizon_metrics"].exists()
    assert paths["failure_roc_auc_plot"].exists()
    assert paths["local_horizon_roc_auc_plot"].exists()

    assert paths["failure_roc_auc_plot"].stat().st_size > 0
    assert paths["local_horizon_roc_auc_plot"].stat().st_size > 0

    global_metrics = pd.read_json(
        paths["global_metrics"],
        typ="series",
    )

    failure_metrics = pd.read_csv(
        paths["failure_metrics"]
    )

    local_metrics = pd.read_csv(
        paths["local_horizon_metrics"]
    )

    assert global_metrics["roc_auc"] == pytest.approx(0.95)
    assert list(failure_metrics["failure_id"]) == [1, 2]
    assert local_metrics.loc[0, "period"] == "1-0h"

def test_run_anomaly_experiment_saves_results(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        experiment_module.pd,
        "read_parquet",
        lambda path: pd.DataFrame({"x": [1, 2]}),
    )

    metadata = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-01-01 00:10"),
                pd.Timestamp("2020-01-02 00:10"),
            ],
        }
    )

    monkeypatch.setattr(
        experiment_module,
        "prepare_model_input",
        lambda df: (
            pd.DataFrame(
                {
                    "feature": [0.1, 0.2],
                }
            ),
            metadata,
        ),
    )

    monkeypatch.setattr(
        experiment_module,
        "train_isolation_forest",
        lambda *args, **kwargs: object(),
    )

    monkeypatch.setattr(
        experiment_module,
        "score_anomalies",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "anomaly_score": [0.1, 0.9],
                "is_anomaly": [False, True],
            }
        ),
    )

    monkeypatch.setattr(
        experiment_module,
        "label_failure_windows",
        lambda *args, **kwargs: pd.DataFrame(
            {
                **metadata,
                "failure_overlap_ratio": [0.0, 1.0],
                "failure_id": pd.Series(
                    [pd.NA, 1],
                    dtype="Int64",
                ),
                "is_failure": [False, True],
            }
        ),
    )

    monkeypatch.setattr(
        experiment_module,
        "get_failure_intervals",
        lambda: pd.DataFrame(),
    )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_global_scores",
        lambda results: {
            "roc_auc": 1.0,
            "pr_auc": 1.0,
        },
    )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_scores_by_failure",
        lambda results: pd.DataFrame(
            {
                "failure_id": [1],
                "roc_auc": [1.0],
            }
        ),
    )

    monkeypatch.setattr(
        experiment_module,
        "evaluate_local_horizons",
        lambda results, failures: pd.DataFrame(
            {
                "failure_id": [1],
                "period": ["durante"],
                "roc_auc": [1.0],
            }
        ),
    )

    report = experiment_module.run_anomaly_experiment(
        "train.parquet",
        "evaluation.parquet",
        output_dir=tmp_path,
    )

    assert "output_files" in report

    assert (
        report["output_files"]["global_metrics"].exists()
    )
    assert (
        report["output_files"]["failure_metrics"].exists()
    )
    assert (
        report["output_files"]["local_horizon_metrics"].exists()
    )
    assert (
    report["output_files"]["failure_roc_auc_plot"].exists()
    )
    assert (
        report["output_files"]["local_horizon_roc_auc_plot"].exists()
    )
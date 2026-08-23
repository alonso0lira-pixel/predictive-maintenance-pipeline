import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.evaluation import (
    evaluate_global_scores,
    evaluate_scores_by_failure,
    evaluate_local_horizons
)


def test_evaluate_global_scores_perfect_ranking() -> None:
    results = pd.DataFrame(
        {
            "is_failure": [
                False,
                False,
                True,
                True,
            ],
            "anomaly_score": [
                0.1,
                0.2,
                0.8,
                0.9,
            ],
        }
    )

    metrics = evaluate_global_scores(results)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_evaluate_global_scores_reports_class_distribution() -> None:
    results = pd.DataFrame(
        {
            "is_failure": [
                False,
                False,
                False,
                True,
            ],
            "anomaly_score": [
                0.1,
                0.2,
                0.3,
                0.9,
            ],
        }
    )

    metrics = evaluate_global_scores(results)

    assert metrics["rows"] == 4
    assert metrics["positives"] == 1
    assert metrics["negatives"] == 3
    assert metrics["prevalence"] == pytest.approx(0.25)


def test_evaluate_global_scores_rejects_missing_columns() -> None:
    results = pd.DataFrame(
        {
            "anomaly_score": [
                0.1,
                0.9,
            ],
        }
    )

    with pytest.raises(
        KeyError,
        match="Faltan columnas necesarias",
    ):
        evaluate_global_scores(results)


def test_evaluate_global_scores_requires_both_classes() -> None:
    results = pd.DataFrame(
        {
            "is_failure": [
                False,
                False,
            ],
            "anomaly_score": [
                0.1,
                0.2,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="cada clase",
    ):
        evaluate_global_scores(results)


@pytest.mark.parametrize(
    "invalid_score",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_evaluate_global_scores_rejects_non_finite_scores(
    invalid_score: float,
) -> None:
    results = pd.DataFrame(
        {
            "is_failure": [
                False,
                True,
            ],
            "anomaly_score": [
                0.1,
                invalid_score,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="no finitos",
    ):
        evaluate_global_scores(results)

def test_evaluate_scores_by_failure() -> None:
    results = pd.DataFrame(
        {
            "failure_id": [
                pd.NA,
                pd.NA,
                1,
                1,
                2,
                2,
            ],
            "is_failure": [
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            "anomaly_score": [
                0.1,
                0.2,
                0.8,
                0.9,
                0.7,
                0.95,
            ],
        }
    )

    metrics = evaluate_scores_by_failure(results)

    assert list(metrics["failure_id"]) == [1, 2]
    assert list(metrics["failure_windows"]) == [2, 2]

    assert metrics.loc[0, "roc_auc"] == pytest.approx(1.0)
    assert metrics.loc[1, "roc_auc"] == pytest.approx(1.0)


def test_evaluate_scores_by_failure_excludes_other_failures() -> None:
    results = pd.DataFrame(
        {
            "failure_id": [
                pd.NA,
                pd.NA,
                1,
                2,
            ],
            "is_failure": [
                False,
                False,
                True,
                True,
            ],
            "anomaly_score": [
                0.1,
                0.2,
                0.8,
                0.9,
            ],
        }
    )

    metrics = evaluate_scores_by_failure(results)

    assert metrics.loc[
        metrics["failure_id"] == 1,
        "roc_auc",
    ].iloc[0] == pytest.approx(1.0)


def test_evaluate_scores_by_failure_requires_background() -> None:
    results = pd.DataFrame(
        {
            "failure_id": [
                1,
                1,
            ],
            "is_failure": [
                True,
                True,
            ],
            "anomaly_score": [
                0.8,
                0.9,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="ventanas no asociadas",
    ):
        evaluate_scores_by_failure(results)
def test_evaluate_scores_by_failure_handles_nullable_failure_id() -> None:
    results = pd.DataFrame(
        {
            "failure_id": pd.Series(
                [
                    pd.NA,
                    pd.NA,
                    1,
                    1,
                    2,
                ],
                dtype="Int64",
            ),
            "is_failure": [
                False,
                False,
                True,
                True,
                True,
            ],
            "anomaly_score": [
                0.1,
                0.2,
                0.8,
                0.9,
                0.85,
            ],
        }
    )

    metrics = evaluate_scores_by_failure(results)

    assert list(metrics["failure_id"]) == [1, 2]
    assert list(metrics["failure_windows"]) == [2, 1]

def test_evaluate_scores_by_failure_respects_failure_label() -> None:
    results = pd.DataFrame(
        {
            "failure_id": pd.Series(
                [
                    pd.NA,
                    pd.NA,
                    1,
                    1,
                ],
                dtype="Int64",
            ),
            "is_failure": [
                False,
                False,
                True,
                False,
            ],
            "anomaly_score": [
                0.1,
                0.2,
                0.9,
                0.8,
            ],
        }
    )

    metrics = evaluate_scores_by_failure(results)

    assert metrics.loc[0, "failure_windows"] == 1

def test_evaluate_local_horizons_detects_separation() -> None:
    base_time = pd.Timestamp("2020-01-10 00:00:00")

    results = pd.DataFrame(
        {
            "window_start_timestamp": [
                base_time - pd.Timedelta(days=3),
                base_time - pd.Timedelta(days=2),
                base_time - pd.Timedelta(hours=11),
                base_time - pd.Timedelta(hours=5),
                base_time + pd.Timedelta(minutes=10),
            ],
            "window_end_timestamp": [
                base_time - pd.Timedelta(days=3) + pd.Timedelta(minutes=10),
                base_time - pd.Timedelta(days=2) + pd.Timedelta(minutes=10),
                base_time - pd.Timedelta(hours=11) + pd.Timedelta(minutes=10),
                base_time - pd.Timedelta(hours=5) + pd.Timedelta(minutes=10),
                base_time + pd.Timedelta(minutes=20),
            ],
            "is_failure": [
                False,
                False,
                False,
                False,
                True,
            ],
            "anomaly_score": [
                0.1,
                0.2,
                0.8,
                0.9,
                0.95,
            ],
        }
    )

    failures = pd.DataFrame(
        {
            "failure_id": [1],
            "start_timestamp": [base_time],
            "end_timestamp": [
                base_time + pd.Timedelta(hours=1)
            ],
        }
    )

    metrics = evaluate_local_horizons(
        results,
        failures,
    )

    twelve_to_six = metrics.loc[
        metrics["period"] == "12-6h"
    ].iloc[0]

    assert twelve_to_six["roc_auc"] == pytest.approx(1.0)
    assert twelve_to_six["delta_mean"] > 0


def test_evaluate_local_horizons_excludes_failures_from_reference() -> None:
    base_time = pd.Timestamp("2020-01-10 00:00:00")

    results = pd.DataFrame(
        {
            "window_start_timestamp": [
                base_time - pd.Timedelta(days=3),
                base_time - pd.Timedelta(days=2),
                base_time - pd.Timedelta(hours=11),
            ],
            "window_end_timestamp": [
                base_time - pd.Timedelta(days=3) + pd.Timedelta(minutes=10),
                base_time - pd.Timedelta(days=2) + pd.Timedelta(minutes=10),
                base_time - pd.Timedelta(hours=11) + pd.Timedelta(minutes=10),
            ],
            "is_failure": [
                False,
                True,
                False,
            ],
            "anomaly_score": [
                0.1,
                0.99,
                0.8,
            ],
        }
    )

    failures = pd.DataFrame(
        {
            "failure_id": [1],
            "start_timestamp": [base_time],
            "end_timestamp": [
                base_time + pd.Timedelta(hours=1)
            ],
        }
    )

    metrics = evaluate_local_horizons(
        results,
        failures,
    )

    twelve_to_six = metrics.loc[
        metrics["period"] == "12-6h"
    ].iloc[0]

    assert twelve_to_six["n_reference"] == 1
    assert twelve_to_six["roc_auc"] == pytest.approx(1.0)


def test_evaluate_local_horizons_rejects_missing_columns() -> None:
    results = pd.DataFrame(
        {
            "anomaly_score": [0.1],
        }
    )

    failures = pd.DataFrame(
        {
            "failure_id": [1],
            "start_timestamp": [pd.Timestamp("2020-01-10")],
            "end_timestamp": [pd.Timestamp("2020-01-11")],
        }
    )

    with pytest.raises(
        KeyError,
        match="Faltan columnas necesarias en results",
    ):
        evaluate_local_horizons(
            results,
            failures,
        )


def test_evaluate_local_horizons_rejects_invalid_failure_interval() -> None:
    results = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-01-05"),
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-01-05 00:10"),
            ],
            "is_failure": [False],
            "anomaly_score": [0.1],
        }
    )

    failures = pd.DataFrame(
        {
            "failure_id": [1],
            "start_timestamp": [pd.Timestamp("2020-01-10")],
            "end_timestamp": [pd.Timestamp("2020-01-09")],
        }
    )

    with pytest.raises(
        ValueError,
        match="Intervalo inválido",
    ):
        evaluate_local_horizons(
            results,
            failures,
        )
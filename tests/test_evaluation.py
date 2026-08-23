import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.evaluation import (
    evaluate_global_scores,
    evaluate_scores_by_failure,
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
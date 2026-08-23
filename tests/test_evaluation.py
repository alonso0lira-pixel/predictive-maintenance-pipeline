import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.evaluation import (
    evaluate_global_scores,
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
"""Evaluación de las puntuaciones de detección de anomalías."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def _validate_binary_labels(
    labels: pd.Series,
) -> pd.Series:
    """Valida etiquetas binarias y las devuelve como booleanos."""

    if labels.isna().any():
        raise ValueError(
            "Las etiquetas contienen valores nulos"
        )

    unique_labels = set(labels.unique())

    if not unique_labels.issubset(
        {False, True, 0, 1}
    ):
        raise ValueError(
            "Las etiquetas deben ser binarias"
        )

    return labels.astype(bool)

def evaluate_global_scores(
    results: pd.DataFrame,
    label_column: str = "is_failure",
    score_column: str = "anomaly_score",
) -> dict[str, float | int]:
    """Calcula métricas globales frente al ground truth documentado."""

    required_columns = {
        label_column,
        score_column,
    }

    missing_columns = sorted(
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para evaluación: "
            f"{missing_columns}"
        )

    if results.empty:
        raise ValueError(
            "No se puede evaluar un DataFrame vacío"
        )

    labels = _validate_binary_labels(
        results[label_column]
    )

    y_true = labels.astype(int)

    if y_true.nunique() != 2:
        raise ValueError(
            "La evaluación requiere al menos una ventana "
            "de cada clase"
        )

    y_score = results[
        score_column
    ].to_numpy(dtype="float64")

    if not np.isfinite(y_score).all():
        raise ValueError(
            "Los anomaly scores contienen valores no finitos"
        )

    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    prevalence = float(positives / len(y_true))

    return {
        "rows": len(results),
        "positives": positives,
        "negatives": negatives,
        "prevalence": prevalence,
        "roc_auc": float(
            roc_auc_score(y_true, y_score)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_score)
        ),
    }

def evaluate_scores_by_failure(
    results: pd.DataFrame,
    failure_id_column: str = "failure_id",
    label_column: str = "is_failure",
    score_column: str = "anomaly_score",
) -> pd.DataFrame:
    """Evalúa separadamente las ventanas asociadas a cada fallo."""

    required_columns = {
        failure_id_column,
        label_column,
        score_column,
    }

    missing_columns = sorted(
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para evaluación por fallo: "
            f"{missing_columns}"
        )

    if results.empty:
        raise ValueError(
            "No se puede evaluar un DataFrame vacío"
        )

    scores = results[score_column].to_numpy(dtype="float64")

    if not np.isfinite(scores).all():
        raise ValueError(
            "Los anomaly scores contienen valores no finitos"
        )

    labels = _validate_binary_labels(
        results[label_column]
    )
    positive_without_failure_id = (
        labels
        & results[failure_id_column].isna()
    )

    if positive_without_failure_id.any():
        raise ValueError(
            "Las ventanas asociadas a fallos "
            "deben tener un failure_id"
        )
    non_failure = ~labels

    if not non_failure.any():
        raise ValueError(
            "La evaluación requiere ventanas no asociadas a fallos"
        )

    failure_ids = (
        results.loc[
            labels,
            failure_id_column,
        ]
        .unique()
    )

    if len(failure_ids) == 0:
        raise ValueError(
            "No existen ventanas asociadas a fallos"
        )

    rows = []

    for failure_id in sorted(failure_ids):
        current_failure = (
            labels
            & results[failure_id_column]
            .eq(failure_id)
            .fillna(False)
        )

        evaluation_mask = (
            non_failure | current_failure
        )

        subset = results.loc[evaluation_mask]

        y_true = current_failure.loc[
            evaluation_mask
        ].astype(int)

        y_score = subset[
            score_column
        ].to_numpy(dtype="float64")

        failure_scores = results.loc[
            current_failure,
            score_column,
        ]

        rows.append(
            {
                "failure_id": int(failure_id),
                "failure_windows": int(
                    current_failure.sum()
                ),
                "roc_auc": float(
                    roc_auc_score(
                        y_true,
                        y_score,
                    )
                ),
                "average_precision": float(
                    average_precision_score(
                        y_true,
                        y_score,
                    )
                ),
                "score_mean": float(
                    failure_scores.mean()
                ),
                "score_median": float(
                    failure_scores.median()
                ),
                "score_max": float(
                    failure_scores.max()
                ),
            }
        )

    return pd.DataFrame(rows)

def evaluate_local_horizons(
    results: pd.DataFrame,
    failure_intervals: pd.DataFrame,
    score_column: str = "anomaly_score",
    label_column: str = "is_failure",
    window_start_column: str = "window_start_timestamp",
    window_end_column: str = "window_end_timestamp",
    failure_start_column: str = "start_timestamp",
    failure_end_column: str = "end_timestamp",
) -> pd.DataFrame:
    """Evalúa la separación local antes y durante cada fallo."""

    required_result_columns = {
        score_column,
        label_column,
        window_start_column,
        window_end_column,
    }

    missing_results = sorted(
        required_result_columns - set(results.columns)
    )

    if missing_results:
        raise KeyError(
            "Faltan columnas necesarias en results: "
            f"{missing_results}"
        )

    required_failure_columns = {
        "failure_id",
        failure_start_column,
        failure_end_column,
    }

    missing_failures = sorted(
        required_failure_columns - set(failure_intervals.columns)
    )

    if missing_failures:
        raise KeyError(
            "Faltan columnas necesarias en failure_intervals: "
            f"{missing_failures}"
        )

    if results.empty:
        raise ValueError(
            "No se puede evaluar un DataFrame vacío"
        )

    scores = results[score_column].to_numpy(dtype="float64")

    if not np.isfinite(scores).all():
        raise ValueError(
            "Los anomaly scores contienen valores no finitos"
        )
    labels = _validate_binary_labels(
            results[label_column]
        )

    data = results.copy()

    data["_window_midpoint"] = (
        data[window_start_column]
        + (
            data[window_end_column]
            - data[window_start_column]
        ) / 2
    )

    periods = [
        ("24-12h", 24, 12),
        ("12-6h", 12, 6),
        ("6-3h", 6, 3),
        ("3-1h", 3, 1),
        ("1-0h", 1, 0),
    ]

    rows = []

    for _, failure in failure_intervals.iterrows():
        failure_id = int(failure["failure_id"])
        failure_start = pd.Timestamp(
            failure[failure_start_column]
        )
        failure_end = pd.Timestamp(
            failure[failure_end_column]
        )

        if failure_end <= failure_start:
            raise ValueError(
                f"Intervalo inválido para failure_id={failure_id}"
            )

        reference_start = (
            failure_start - pd.Timedelta(days=7)
        )
        reference_end = (
            failure_start - pd.Timedelta(hours=24)
        )

        reference_mask = (
            data["_window_midpoint"].ge(reference_start)
            & data["_window_midpoint"].lt(reference_end)
            & ~labels
        )

        reference_scores = data.loc[
            reference_mask,
            score_column,
        ]

        if reference_scores.empty:
            raise ValueError(
                "No existen ventanas de referencia para "
                f"failure_id={failure_id}"
            )

        for period_name, hours_before_start, hours_before_end in periods:
            period_start = (
                failure_start
                - pd.Timedelta(hours=hours_before_start)
            )
            period_end = (
                failure_start
                - pd.Timedelta(hours=hours_before_end)
            )

            period_mask = (
                data["_window_midpoint"].ge(period_start)
                & data["_window_midpoint"].lt(period_end)
            )

            period_scores = data.loc[
                period_mask,
                score_column,
            ]

            if period_scores.empty:
                continue

            y_true = np.concatenate(
                [
                    np.zeros(
                        len(reference_scores),
                        dtype=int,
                    ),
                    np.ones(
                        len(period_scores),
                        dtype=int,
                    ),
                ]
            )

            y_score = np.concatenate(
                [
                    reference_scores.to_numpy(),
                    period_scores.to_numpy(),
                ]
            )

            rows.append(
                {
                    "failure_id": failure_id,
                    "period": period_name,
                    "n_reference": len(reference_scores),
                    "n_period": len(period_scores),
                    "roc_auc": float(
                        roc_auc_score(
                            y_true,
                            y_score,
                        )
                    ),
                    "delta_mean": float(
                        period_scores.mean()
                        - reference_scores.mean()
                    ),
                }
            )

        during_mask = (
            data["_window_midpoint"].ge(failure_start)
            & data["_window_midpoint"].le(failure_end)
        )

        during_scores = data.loc[
            during_mask,
            score_column,
        ]

        if not during_scores.empty:
            y_true = np.concatenate(
                [
                    np.zeros(
                        len(reference_scores),
                        dtype=int,
                    ),
                    np.ones(
                        len(during_scores),
                        dtype=int,
                    ),
                ]
            )

            y_score = np.concatenate(
                [
                    reference_scores.to_numpy(),
                    during_scores.to_numpy(),
                ]
            )

            rows.append(
                {
                    "failure_id": failure_id,
                    "period": "durante",
                    "n_reference": len(reference_scores),
                    "n_period": len(during_scores),
                    "roc_auc": float(
                        roc_auc_score(
                            y_true,
                            y_score,
                        )
                    ),
                    "delta_mean": float(
                        during_scores.mean()
                        - reference_scores.mean()
                    ),
                }
            )

    return pd.DataFrame(rows)
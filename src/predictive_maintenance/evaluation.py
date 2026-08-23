"""Evaluación de las puntuaciones de detección de anomalías."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


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

    labels = results[label_column]

    if labels.isna().any():
        raise ValueError(
            "Las etiquetas contienen valores nulos"
        )

    unique_labels = set(labels.unique())

    if not unique_labels.issubset({False, True, 0, 1}):
        raise ValueError(
            "Las etiquetas deben ser binarias"
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
        "pr_auc": float(
            average_precision_score(y_true, y_score)
        ),
    }
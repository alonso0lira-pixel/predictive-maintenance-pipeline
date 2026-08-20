"""Detección no supervisada de anomalías."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _validate_features(
    features: pd.DataFrame,
) -> None:
    """Valida la matriz de entrada utilizada por el modelo."""

    if features.empty:
        raise ValueError(
            "La matriz de features no puede estar vacía"
        )

    non_numeric_columns = (
        features.select_dtypes(
            exclude="number"
        ).columns.tolist()
    )

    if non_numeric_columns:
        raise TypeError(
            "Todas las features deben ser numéricas: "
            f"{non_numeric_columns}"
        )

    values = features.to_numpy(
        dtype="float64"
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Las features contienen valores no finitos"
        )


def train_isolation_forest(
    features: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
) -> IsolationForest:
    """Entrena un Isolation Forest sobre las features de train."""

    _validate_features(features)

    if n_estimators <= 0:
        raise ValueError(
            "n_estimators debe ser mayor que cero"
        )

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(features)

    return model


def score_anomalies(
    model: IsolationForest,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula puntuaciones y clasificación de anomalía."""

    _validate_features(features)

    anomaly_score = -model.score_samples(
        features
    )

    prediction = model.predict(
        features
    )

    return pd.DataFrame(
        {
            "anomaly_score": anomaly_score,
            "is_anomaly": prediction == -1,
        },
        index=features.index,
    )
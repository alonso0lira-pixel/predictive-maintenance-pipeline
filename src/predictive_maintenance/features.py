"""Generación de features a partir de ventanas temporales."""

from __future__ import annotations

import pandas as pd

from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
)

from predictive_maintenance.windowing import (
    MODEL_FEATURE_COLUMNS,
    extract_window,
)


def aggregate_window_features(
    window: pd.DataFrame,
) -> pd.Series:
    """Resume una ventana temporal en un vector de features."""

    required_columns = set(ANALOG_COLUMNS) | set(BINARY_COLUMNS)

    missing_columns = sorted(
        required_columns - set(window.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para generar features: "
            f"{missing_columns}"
        )

    if window.empty:
        raise ValueError(
            "No se pueden generar features de una ventana vacía"
        )

    features: dict[str, float] = {}

    for column in ANALOG_COLUMNS:
        series = window[column]

        features[f"{column}_mean"] = float(series.mean())
        features[f"{column}_std"] = float(series.std(ddof=0))
        features[f"{column}_min"] = float(series.min())
        features[f"{column}_max"] = float(series.max())

    for column in BINARY_COLUMNS:
        series = window[column]

        features[f"{column}_active_ratio"] = float(
            series.mean()
        )

        transitions = int(
            series.diff()
            .fillna(0)
            .ne(0)
            .sum()
        )

        features[f"{column}_transitions"] = float(
            transitions
        )

    return pd.Series(features, dtype="float64")

def build_feature_dataset(
    df: pd.DataFrame,
    window_index: pd.DataFrame,
) -> pd.DataFrame:
    """Construye una tabla de features a partir del índice de ventanas."""

    required_index_columns = {
        "segment_id",
        "start_index",
        "end_index",
    }

    missing_index_columns = sorted(
        required_index_columns - set(window_index.columns)
    )

    if missing_index_columns:
        raise KeyError(
            "Faltan columnas necesarias en el índice de ventanas: "
            f"{missing_index_columns}"
        )

    rows: list[dict[str, object]] = []

    for window_info in window_index.itertuples(index=False):
        window = extract_window(
            df,
            start_index=int(window_info.start_index),
            end_index=int(window_info.end_index),
            feature_columns=MODEL_FEATURE_COLUMNS,
        )

        aggregated = aggregate_window_features(window)

        row: dict[str, object] = {
            "segment_id": int(window_info.segment_id),
            "start_index": int(window_info.start_index),
            "end_index": int(window_info.end_index),
        }

        row.update(aggregated.to_dict())

        rows.append(row)

    return pd.DataFrame(rows)
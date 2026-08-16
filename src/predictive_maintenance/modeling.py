"""Preparación de los datasets para modelado."""

from __future__ import annotations

import pandas as pd

from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
)


MODEL_METADATA_COLUMNS = [
    "segment_id",
    "start_index",
    "end_index",
    "window_start_timestamp",
    "window_end_timestamp",
]

MODEL_INPUT_COLUMNS = [
    *[
        f"{column}_{statistic}"
        for column in ANALOG_COLUMNS
        for statistic in (
            "mean",
            "std",
            "min",
            "max",
        )
    ],
    *[
        f"{column}_{statistic}"
        for column in BINARY_COLUMNS
        for statistic in (
            "active_ratio",
            "transitions",
        )
    ],
]


def prepare_model_input(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa las features del modelo de sus metadatos."""

    required_columns = (
        MODEL_METADATA_COLUMNS
        + MODEL_INPUT_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in feature_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para modelado: "
            f"{missing_columns}"
        )

    features = feature_df.loc[
        :,
        MODEL_INPUT_COLUMNS,
    ].copy()

    metadata = feature_df.loc[
        :,
        MODEL_METADATA_COLUMNS,
    ].copy()

    return features, metadata
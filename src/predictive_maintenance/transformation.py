"""Transformaciones aplicadas al dataset MetroPT-3."""

from __future__ import annotations

import pandas as pd


AUXILIARY_COLUMNS = ["Unnamed: 0"]


def remove_auxiliary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas auxiliares sin modificar el DataFrame original."""

    missing_columns = [
        column
        for column in AUXILIARY_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "No se encontraron las columnas auxiliares esperadas: "
            f"{missing_columns}"
        )

    return df.drop(columns=AUXILIARY_COLUMNS)

def normalize_timestamp(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Convierte la columna temporal a datetime sin modificar el original."""

    if timestamp_column not in df.columns:
        raise KeyError(
            f"No se encontró la columna temporal: {timestamp_column}"
        )

    transformed_df = df.copy()

    parsed_timestamps = pd.to_datetime(
        transformed_df[timestamp_column],
        errors="coerce",
    )

    invalid_timestamps = int(parsed_timestamps.isna().sum())

    if invalid_timestamps > 0:
        raise ValueError(
            f"La columna '{timestamp_column}' contiene "
            f"{invalid_timestamps} valores temporales no interpretables"
        )

    transformed_df[timestamp_column] = parsed_timestamps

    return transformed_df

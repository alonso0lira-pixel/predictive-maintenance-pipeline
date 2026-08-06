"""Transformaciones aplicadas al dataset MetroPT-3."""

from __future__ import annotations

import pandas as pd

from predictive_maintenance.validation import (
    ALLOWED_BINARY_VALUES,
    BINARY_COLUMNS,
)


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


def cast_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte las señales binarias a int8 sin modificar el original."""

    missing_columns = sorted(
        set(BINARY_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            "No se encontraron las columnas binarias esperadas: "
            f"{missing_columns}"
        )

    transformed_df = df.copy()

    for column in BINARY_COLUMNS:
        if transformed_df[column].isna().any():
            raise ValueError(
                f"La columna binaria '{column}' contiene valores nulos"
            )

        invalid_values = sorted(
            set(transformed_df[column].unique())
            - ALLOWED_BINARY_VALUES,
            key=str,
        )

        if invalid_values:
            raise ValueError(
                f"La columna binaria '{column}' contiene "
                f"valores no permitidos: {invalid_values}"
            )

        transformed_df[column] = transformed_df[column].astype(
            "int8"
        )

    return transformed_df


def transform_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta todas las transformaciones definidas para MetroPT-3."""

    transformed_df = remove_auxiliary_columns(df)
    transformed_df = normalize_timestamp(transformed_df)
    transformed_df = cast_binary_columns(transformed_df)

    return transformed_df
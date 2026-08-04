"""Validaciones de calidad para el dataset MetroPT-3."""

from __future__ import annotations

import pandas as pd


EXPECTED_COLUMNS = {
    "Unnamed: 0",
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
}

ANALOG_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

BINARY_COLUMNS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

ALLOWED_BINARY_VALUES = {0, 1}

EXPECTED_INTERVAL = pd.Timedelta(seconds=10)
NORMAL_MAX_INTERVAL = pd.Timedelta(seconds=13)

def validate_schema(df: pd.DataFrame) -> dict[str, object]:
    """Comprueba que el DataFrame contiene exactamente las columnas esperadas."""

    actual_columns = set(df.columns)

    missing_columns = sorted(EXPECTED_COLUMNS - actual_columns)
    unexpected_columns = sorted(actual_columns - EXPECTED_COLUMNS)

    is_valid = not missing_columns and not unexpected_columns

    return {
        "is_valid": is_valid,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
    }

def validate_missing_values(df: pd.DataFrame) -> dict[str, object]:
    """Comprueba si el DataFrame contiene valores nulos."""

    missing_counts = df.isna().sum()

    missing_by_column = {
        column: int(count)
        for column, count in missing_counts.items()
        if count > 0
    }

    total_missing_values = sum(missing_by_column.values())

    return {
        "is_valid": total_missing_values == 0,
        "total_missing_values": total_missing_values,
        "missing_by_column": missing_by_column,
    }

def validate_duplicate_rows(df: pd.DataFrame) -> dict[str, object]:
    """Comprueba si el DataFrame contiene filas completamente duplicadas."""

    duplicate_rows = int(df.duplicated().sum())

    return {
        "is_valid": duplicate_rows == 0,
        "duplicate_rows": duplicate_rows,
    }

def validate_duplicate_timestamps(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> dict[str, object]:
    """Comprueba si existen timestamps repetidos."""

    duplicate_timestamps = int(
        df[timestamp_column].duplicated().sum()
    )

    return {
        "is_valid": duplicate_timestamps == 0,
        "duplicate_timestamps": duplicate_timestamps,
    }

def validate_chronological_order(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> dict[str, object]:
    """Comprueba que los timestamps estén ordenados cronológicamente."""

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    invalid_timestamps = int(timestamps.isna().sum())

    temporal_reversals = int(
        timestamps.diff().lt(pd.Timedelta(0)).sum()
    )

    return {
        "is_valid": (
            invalid_timestamps == 0
            and temporal_reversals == 0
        ),
        "invalid_timestamps": invalid_timestamps,
        "temporal_reversals": temporal_reversals,
    }

def validate_binary_values(df: pd.DataFrame) -> dict[str, object]:
    """Comprueba que las señales binarias contengan únicamente 0 y 1."""

    missing_columns = sorted(
        set(BINARY_COLUMNS) - set(df.columns)
    )

    invalid_values_by_column: dict[str, list[object]] = {}
    invalid_counts_by_column: dict[str, int] = {}

    for column in BINARY_COLUMNS:
        if column not in df.columns:
            continue

        non_null_values = df[column].dropna()

        invalid_mask = ~non_null_values.isin(
            ALLOWED_BINARY_VALUES
        )

        if invalid_mask.any():
            invalid_values = (
                non_null_values.loc[invalid_mask]
                .unique()
                .tolist()
            )

            invalid_values_by_column[column] = sorted(
                invalid_values,
                key=str,
            )

            invalid_counts_by_column[column] = int(
                invalid_mask.sum()
            )

    total_invalid_values = sum(
        invalid_counts_by_column.values()
    )

    return {
        "is_valid": (
            not missing_columns
            and total_invalid_values == 0
        ),
        "missing_columns": missing_columns,
        "total_invalid_values": total_invalid_values,
        "invalid_counts_by_column": invalid_counts_by_column,
        "invalid_values_by_column": invalid_values_by_column,
    }
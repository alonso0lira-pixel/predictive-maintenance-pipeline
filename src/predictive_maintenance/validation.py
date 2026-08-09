"""Validaciones de calidad para el dataset MetroPT-3."""
from __future__ import annotations


import numpy as np
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


def validate_analog_values(df: pd.DataFrame) -> dict[str, object]:
    """Valida la estructura y los valores de las señales analógicas."""

    missing_columns = sorted(
        set(ANALOG_COLUMNS) - set(df.columns)
    )

    non_numeric_columns: list[str] = []
    constant_columns: list[str] = []
    infinite_counts_by_column: dict[str, int] = {}

    for column in ANALOG_COLUMNS:
        if column not in df.columns:
            continue

        series = df[column]

        if not pd.api.types.is_numeric_dtype(series):
            non_numeric_columns.append(column)
            continue

        non_null_values = series.dropna()

        infinite_mask = np.isinf(
            non_null_values.to_numpy(dtype=float)
        )

        infinite_count = int(infinite_mask.sum())

        if infinite_count > 0:
            infinite_counts_by_column[column] = infinite_count

        finite_values = non_null_values.loc[~infinite_mask]

        if finite_values.nunique() <= 1:
            constant_columns.append(column)

    total_infinite_values = sum(
        infinite_counts_by_column.values()
    )

    return {
        "is_valid": (
            not missing_columns
            and not non_numeric_columns
            and not constant_columns
            and total_infinite_values == 0
        ),
        "missing_columns": missing_columns,
        "non_numeric_columns": sorted(non_numeric_columns),
        "constant_columns": sorted(constant_columns),
        "total_infinite_values": total_infinite_values,
        "infinite_counts_by_column": infinite_counts_by_column,
    }

def validate_temporal_gaps(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    normal_max_interval: pd.Timedelta = NORMAL_MAX_INTERVAL,
) -> dict[str, object]:
    """Detecta interrupciones superiores al intervalo temporal normal."""

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    invalid_timestamps = int(timestamps.isna().sum())

    intervals = timestamps.diff()
    positive_intervals = intervals.loc[
        intervals > pd.Timedelta(0)
    ]

    total_gaps = int(
        (positive_intervals > normal_max_interval).sum()
    )

    gaps_over_1_minute = int(
        (positive_intervals > pd.Timedelta(minutes=1)).sum()
    )

    gaps_over_5_minutes = int(
        (positive_intervals > pd.Timedelta(minutes=5)).sum()
    )

    gaps_over_1_hour = int(
        (positive_intervals > pd.Timedelta(hours=1)).sum()
    )

    maximum_interval = positive_intervals.max()

    max_gap_seconds = (
        None
        if pd.isna(maximum_interval)
        else float(maximum_interval.total_seconds())
    )

    return {
        "is_valid": invalid_timestamps == 0,
        "has_warning": total_gaps > 0,
        "invalid_timestamps": invalid_timestamps,
        "total_gaps": total_gaps,
        "gaps_over_1_minute": gaps_over_1_minute,
        "gaps_over_5_minutes": gaps_over_5_minutes,
        "gaps_over_1_hour": gaps_over_1_hour,
        "max_gap_seconds": max_gap_seconds,
    }


def validate_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Ejecuta el conjunto completo de validaciones del dataset."""

    checks: dict[str, dict[str, object]] = {
        "schema": validate_schema(df),
        "missing_values": validate_missing_values(df),
        "duplicate_rows": validate_duplicate_rows(df),
        "binary_values": validate_binary_values(df),
        "analog_values": validate_analog_values(df),
    }

    skipped_checks: list[str] = []

    if "timestamp" in df.columns:
        checks["duplicate_timestamps"] = (
            validate_duplicate_timestamps(df)
        )
        checks["chronological_order"] = (
            validate_chronological_order(df)
        )
        checks["temporal_gaps"] = validate_temporal_gaps(df)
    else:
        skipped_checks.extend(
            [
                "duplicate_timestamps",
                "chronological_order",
                "temporal_gaps",
            ]
        )

    is_valid = all(
        bool(result["is_valid"])
        for result in checks.values()
    )

    has_warnings = bool(
        checks.get("temporal_gaps", {}).get(
            "has_warning",
            False,
        )
    )

    return {
        "is_valid": is_valid,
        "has_warnings": has_warnings,
        "checks": checks,
        "skipped_checks": skipped_checks,
    }
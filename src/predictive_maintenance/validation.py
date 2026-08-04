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

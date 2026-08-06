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

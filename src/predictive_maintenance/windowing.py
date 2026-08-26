"""Utilidades para construir ventanas temporales."""

from __future__ import annotations

import pandas as pd

from collections.abc import Sequence

from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
)

MODEL_FEATURE_COLUMNS = [
    *ANALOG_COLUMNS,
    *BINARY_COLUMNS,
]

def _validate_contiguous_segments(
    df: pd.DataFrame,
    segment_column: str,
) -> None:
    """Comprueba que cada segmento aparezca en un único bloque continuo."""

    if df[segment_column].isna().any():
        raise ValueError(
            f"La columna {segment_column} contiene valores nulos"
        )

    segment_runs = df.loc[
        df[segment_column].ne(
            df[segment_column].shift()
        ),
        segment_column,
    ]

    duplicated_runs = segment_runs[
        segment_runs.duplicated()
    ]

    if not duplicated_runs.empty:
        repeated_segments = sorted(
            duplicated_runs.unique().tolist()
        )

        raise ValueError(
            "Los segmentos deben ocupar bloques contiguos. "
            "Segmentos intercalados detectados: "
            f"{repeated_segments}"
        )

def count_windows_by_segment(
    df: pd.DataFrame,
    window_size: int,
    step_size: int = 1,
    segment_column: str = "segment_id",
) -> pd.DataFrame:
    """Calcula cuántas ventanas pueden generarse dentro de cada segmento."""

    if window_size <= 0:
        raise ValueError("window_size debe ser mayor que cero")

    if step_size <= 0:
        raise ValueError("step_size debe ser mayor que cero")

    if segment_column not in df.columns:
        raise KeyError(
            f"No se encontró la columna de segmento: {segment_column}"
        )


    segment_sizes = (
        df.groupby(segment_column, sort=True)
        .size()
        .rename("rows")
    )

    number_of_windows = (
        ((segment_sizes - window_size) // step_size) + 1
    ).clip(lower=0)

    result = pd.DataFrame(
        {
            segment_column: segment_sizes.index,
            "rows": segment_sizes.values,
            "windows": number_of_windows.values,
        }
    )

    return result

def generate_window_index(
    df: pd.DataFrame,
    window_size: int = 60,
    step_size: int = 30,
    segment_column: str = "segment_id",
) -> pd.DataFrame:
    """Genera los límites de cada ventana sin materializar sus datos."""

    if window_size <= 0:
        raise ValueError("window_size debe ser mayor que cero")

    if step_size <= 0:
        raise ValueError("step_size debe ser mayor que cero")

    if segment_column not in df.columns:
        raise KeyError(
            f"No se encontró la columna de segmento: {segment_column}"
        )
    
    _validate_contiguous_segments(
        df,
        segment_column,
    )

    windows: list[dict[str, int]] = []

    working_df = df.reset_index(drop=True)

    for segment_id, segment_df in working_df.groupby(
        segment_column,
        sort=True,
    ):
        segment_length = len(segment_df)

        if segment_length < window_size:
            continue

        segment_start = int(segment_df.index[0])

        for local_start in range(
            0,
            segment_length - window_size + 1,
            step_size,
        ):
            start_index = segment_start + local_start
            end_index = start_index + window_size

            windows.append(
                {
                    "segment_id": int(segment_id),
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )

    

    return pd.DataFrame(
        windows,
        columns=[
            "segment_id",
            "start_index",
            "end_index",
        ],
    )

def extract_window(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Extrae una ventana por posición conservando únicamente las features."""

    if start_index < 0:
        raise ValueError("start_index no puede ser negativo")

    if end_index <= start_index:
        raise ValueError("end_index debe ser mayor que start_index")

    if end_index > len(df):
        raise IndexError(
            "La ventana solicitada supera el tamaño del DataFrame"
        )

    missing_columns = sorted(
        set(feature_columns) - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para la ventana: "
            f"{missing_columns}"
        )

    return (
        df.iloc[start_index:end_index]
        .loc[:, list(feature_columns)]
        .copy()
    )
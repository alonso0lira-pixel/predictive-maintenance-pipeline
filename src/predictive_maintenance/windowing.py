"""Utilidades para construir ventanas temporales."""

from __future__ import annotations

import pandas as pd


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
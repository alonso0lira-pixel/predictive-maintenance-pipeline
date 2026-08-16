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

    windows: list[dict[str, int]] = []

    for segment_id, segment_df in df.groupby(
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
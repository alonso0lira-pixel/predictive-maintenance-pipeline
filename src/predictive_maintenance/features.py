"""Generación de features a partir de ventanas temporales."""

from __future__ import annotations

import pandas as pd
import numpy as np

from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
)

from predictive_maintenance.windowing import (
    MODEL_FEATURE_COLUMNS,
    extract_window,
)


def aggregate_window_features(
    window: pd.DataFrame,
) -> pd.Series:
    """Resume una ventana temporal en un vector de features."""

    required_columns = set(ANALOG_COLUMNS) | set(BINARY_COLUMNS)

    missing_columns = sorted(
        required_columns - set(window.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para generar features: "
            f"{missing_columns}"
        )

    if window.empty:
        raise ValueError(
            "No se pueden generar features de una ventana vacía"
        )

    features: dict[str, float] = {}

    for column in ANALOG_COLUMNS:
        series = window[column]

        features[f"{column}_mean"] = float(series.mean())
        features[f"{column}_std"] = float(series.std(ddof=0))
        features[f"{column}_min"] = float(series.min())
        features[f"{column}_max"] = float(series.max())

    for column in BINARY_COLUMNS:
        series = window[column]

        features[f"{column}_active_ratio"] = float(
            series.mean()
        )

        transitions = int(
            series.diff()
            .fillna(0)
            .ne(0)
            .sum()
        )

        features[f"{column}_transitions"] = float(
            transitions
        )

    return pd.Series(features, dtype="float64")

def build_feature_dataset(
    df: pd.DataFrame,
    window_index: pd.DataFrame,
) -> pd.DataFrame:
    """Construye features vectorizadas a partir del índice de ventanas."""

    required_index_columns = {
        "segment_id",
        "start_index",
        "end_index",
    }

    missing_index_columns = sorted(
        required_index_columns - set(window_index.columns)
    )

    if missing_index_columns:
        raise KeyError(
            "Faltan columnas necesarias en el índice de ventanas: "
            f"{missing_index_columns}"
        )

    if window_index.empty:
        return pd.DataFrame()

    working_df = df.reset_index(drop=True)

    if "timestamp" not in working_df.columns:
        raise KeyError(
            "El dataset debe contener la columna timestamp "
            "para mantener la trazabilidad temporal"
        )

    result_chunks: list[pd.DataFrame] = []

    for segment_id, segment_windows in window_index.groupby(
        "segment_id",
        sort=False,
    ):
        window_sizes = (
            segment_windows["end_index"]
            - segment_windows["start_index"]
        )

        if window_sizes.nunique() != 1:
            raise ValueError(
                "Todas las ventanas de un segmento deben "
                "tener el mismo tamaño"
            )

        window_size = int(window_sizes.iloc[0])

        segment_start = int(
            segment_windows["start_index"].min()
        )

        segment_end = int(
            segment_windows["end_index"].max()
        )

        if segment_start < 0 or segment_end > len(working_df):
            raise IndexError(
                "El índice de ventanas contiene límites "
                "fuera del DataFrame"
            )

        start_positions = (
            segment_windows["start_index"]
            .to_numpy(dtype="int64")
        )

        end_positions = (
            segment_windows["end_index"]
            .to_numpy(dtype="int64")
        )

        local_starts = (
            start_positions
            - segment_start
        )

        segment_df = working_df.iloc[
            segment_start:segment_end
        ]

        analog_values = segment_df[
            ANALOG_COLUMNS
        ].to_numpy(dtype="float64")

        binary_values = segment_df[
            BINARY_COLUMNS
        ].to_numpy(dtype="int8")

        analog_windows = np.lib.stride_tricks.sliding_window_view(
            analog_values,
            window_shape=window_size,
            axis=0,
        )[local_starts]

        binary_windows = np.lib.stride_tricks.sliding_window_view(
            binary_values,
            window_shape=window_size,
            axis=0,
        )[local_starts]

        analog_mean = analog_windows.mean(axis=2)
        analog_std = analog_windows.std(axis=2)
        analog_min = analog_windows.min(axis=2)
        analog_max = analog_windows.max(axis=2)

        binary_active_ratio = binary_windows.mean(axis=2)

        binary_transitions = np.count_nonzero(
            np.diff(binary_windows, axis=2),
            axis=2,
        )

        chunk_data: dict[str, object] = {
            "segment_id": segment_windows[
                "segment_id"
            ].to_numpy(),
            "start_index": start_positions,
            "end_index": end_positions,
            "window_start_timestamp": (
                working_df.iloc[start_positions]["timestamp"]
                .to_numpy()
            ),
            "window_end_timestamp": (
                working_df.iloc[end_positions - 1]["timestamp"]
                .to_numpy()
            ),
        }

        for i, column in enumerate(ANALOG_COLUMNS):
            chunk_data[f"{column}_mean"] = analog_mean[:, i]
            chunk_data[f"{column}_std"] = analog_std[:, i]
            chunk_data[f"{column}_min"] = analog_min[:, i]
            chunk_data[f"{column}_max"] = analog_max[:, i]

        for i, column in enumerate(BINARY_COLUMNS):
            chunk_data[
                f"{column}_active_ratio"
            ] = binary_active_ratio[:, i]

            chunk_data[
                f"{column}_transitions"
            ] = binary_transitions[:, i].astype(
                "float64"
            )

        result_chunks.append(
            pd.DataFrame(chunk_data)
        )

    return pd.concat(
        result_chunks,
        ignore_index=True,
    )
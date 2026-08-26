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

def _validate_window_index_against_source(
    df: pd.DataFrame,
    window_index: pd.DataFrame,
) -> None:
    """Valida que las ventanas sean coherentes con el dataset fuente."""

    if "segment_id" not in df.columns:
        raise KeyError(
            "El dataset debe contener la columna segment_id"
        )

    if df["segment_id"].isna().any():
        raise ValueError(
            "La columna segment_id contiene valores nulos"
        )

    for column in (
        "segment_id",
        "start_index",
        "end_index",
    ):
        if window_index[column].isna().any():
            raise ValueError(
                f"La columna {column} del índice de ventanas "
                "contiene valores nulos"
            )

    for column in (
        "start_index",
        "end_index",
    ):
        if not pd.api.types.is_integer_dtype(
            window_index[column]
        ):
            raise TypeError(
                f"La columna {column} debe contener enteros"
            )

    start_positions = (
        window_index["start_index"]
        .to_numpy(dtype="int64")
    )

    end_positions = (
        window_index["end_index"]
        .to_numpy(dtype="int64")
    )

    if (start_positions < 0).any():
        raise ValueError(
            "start_index no puede contener valores negativos"
        )

    if (end_positions <= start_positions).any():
        raise ValueError(
            "Cada ventana debe tener end_index > start_index"
        )

    if (end_positions > len(df)).any():
        raise IndexError(
            "El índice de ventanas contiene límites "
            "fuera del DataFrame"
        )

    window_sizes = (
        end_positions - start_positions
    )

    if np.unique(window_sizes).size != 1:
        raise ValueError(
            "Todas las ventanas deben tener el mismo tamaño"
        )

    source_segments = (
        df["segment_id"]
        .reset_index(drop=True)
        .to_numpy()
    )

    declared_segments = (
        window_index["segment_id"]
        .to_numpy()
    )

    if not np.array_equal(
        source_segments[start_positions],
        declared_segments,
    ):
        raise ValueError(
            "El segment_id del índice de ventanas "
            "no coincide con el dataset fuente"
        )

    # Identifica los bloques temporales reales del dataset.
    segment_changes = np.zeros(
        len(source_segments),
        dtype="int64",
    )

    if len(source_segments) > 1:
        segment_changes[1:] = (
            source_segments[1:]
            != source_segments[:-1]
        )

    segment_runs = np.cumsum(
        segment_changes
    )

    if (
        segment_runs[start_positions]
        != segment_runs[end_positions - 1]
    ).any():
        raise ValueError(
            "Una o más ventanas atraviesan "
            "límites entre segmentos"
        )

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
    _validate_window_index_against_source(
        working_df,
        window_index,
    )

    ordered_window_index = (
        window_index
        .reset_index(drop=True)
        .copy()
    )

    ordered_window_index["_window_order"] = np.arange(
        len(ordered_window_index)
    )

    result_chunks: list[pd.DataFrame] = []

    for segment_id, segment_windows in ordered_window_index.groupby(
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
            "_window_order": segment_windows[
                "_window_order"
            ].to_numpy(),
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

    result = pd.concat(
            result_chunks,
            ignore_index=True,
        )

    return (
        result
        .sort_values("_window_order")
        .drop(columns="_window_order")
        .reset_index(drop=True)
    )
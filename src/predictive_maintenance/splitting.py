"""División temporal del dataset evitando fuga entre segmentos."""

from __future__ import annotations

import pandas as pd


def temporal_split_by_segment(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    segment_column: str = "segment_id",
) -> dict[str, pd.DataFrame]:
    """Divide cronológicamente el dataset manteniendo segmentos completos."""

    if df.empty:
        raise ValueError("No se puede dividir un DataFrame vacío")

    if segment_column not in df.columns:
        raise KeyError(
            f"No existe la columna de segmento: {segment_column}"
        )

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio debe estar entre 0 y 1")

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio debe estar entre 0 y 1"
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio debe ser menor que 1"
        )

    if df[segment_column].isna().any():
        raise ValueError(
            "La columna de segmento contiene valores nulos"
        )

    segment_sizes = (
        df.groupby(segment_column, sort=False)
        .size()
    )

    number_of_segments = len(segment_sizes)

    if number_of_segments < 3:
        raise ValueError(
            "Se necesitan al menos tres segmentos para "
            "crear train, validation y test"
        )

    # Cada segment_id debe aparecer en un único bloque continuo.
    segment_runs = (
        df[segment_column]
        .ne(df[segment_column].shift())
        .sum()
    )

    if segment_runs != number_of_segments:
        raise ValueError(
            "Los segmentos deben aparecer en bloques continuos"
        )

    cumulative_rows = segment_sizes.cumsum().to_numpy()

    train_target = len(df) * train_ratio
    validation_target = len(df) * (
        train_ratio + validation_ratio
    )

    train_end = (
        int(
            cumulative_rows.searchsorted(
                train_target,
                side="left",
            )
        )
        + 1
    )

    validation_end = (
        int(
            cumulative_rows.searchsorted(
                validation_target,
                side="left",
            )
        )
        + 1
    )

    # Garantizamos al menos un segmento en cada conjunto.
    train_end = min(
        max(train_end, 1),
        number_of_segments - 2,
    )

    validation_end = min(
        max(validation_end, train_end + 1),
        number_of_segments - 1,
    )

    segment_ids = segment_sizes.index.tolist()

    train_segments = set(
        segment_ids[:train_end]
    )
    validation_segments = set(
        segment_ids[train_end:validation_end]
    )
    test_segments = set(
        segment_ids[validation_end:]
    )

    return {
        "train": df[
            df[segment_column].isin(train_segments)
        ].copy(),
        "validation": df[
            df[segment_column].isin(validation_segments)
        ].copy(),
        "test": df[
            df[segment_column].isin(test_segments)
        ].copy(),
    }
def temporal_split_by_cutoff(
    df: pd.DataFrame,
    cutoff_timestamp: str | pd.Timestamp,
    timestamp_column: str = "timestamp",
    segment_column: str = "segment_id",
) -> dict[str, pd.DataFrame]:
    """Divide el dataset en train y evaluación usando un corte temporal."""

    if df.empty:
        raise ValueError(
            "No se puede dividir un DataFrame vacío"
        )

    required_columns = {
        timestamp_column,
        segment_column,
    }

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para el split temporal: "
            f"{missing_columns}"
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df[timestamp_column]
    ):
        raise TypeError(
            "La columna temporal debe tener tipo datetime"
        )

    if df[timestamp_column].isna().any():
        raise ValueError(
            "La columna temporal contiene valores nulos"
        )

    if not df[timestamp_column].is_monotonic_increasing:
        raise ValueError(
            "El dataset debe estar ordenado cronológicamente"
        )

    cutoff = pd.Timestamp(
        cutoff_timestamp
    )

    train = df[
        df[timestamp_column] < cutoff
    ].copy()

    evaluation = df[
        df[timestamp_column] >= cutoff
    ].copy()

    if train.empty or evaluation.empty:
        raise ValueError(
            "El corte temporal debe dejar datos "
            "tanto en train como en evaluación"
        )

    return {
        "train": train,
        "evaluation": evaluation,
    }
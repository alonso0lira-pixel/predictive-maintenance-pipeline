"""Etiquetado temporal de ventanas mediante eventos de fallo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_maintenance.ground_truth import (
    get_failure_intervals,
)


def label_failure_windows(
    windows: pd.DataFrame,
    overlap_threshold: float = 0.50,
) -> pd.DataFrame:
    """Añade información de solapamiento con fallos documentados."""

    required_columns = {
        "window_start_timestamp",
        "window_end_timestamp",
    }

    missing_columns = sorted(
        required_columns - set(windows.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas temporales necesarias: "
            f"{missing_columns}"
        )

    if not 0 < overlap_threshold <= 1:
        raise ValueError(
            "overlap_threshold debe estar entre 0 y 1"
        )

    result = windows.copy()

    if result.empty:
        result["failure_overlap_ratio"] = pd.Series(
            dtype="float64"
        )
        result["failure_id"] = pd.Series(
            dtype="Int64"
        )
        result["is_failure"] = pd.Series(
            dtype="bool"
        )
        return result

    window_duration = (
        result["window_end_timestamp"]
        - result["window_start_timestamp"]
    ).dt.total_seconds()

    if (window_duration <= 0).any():
        raise ValueError(
            "Todas las ventanas deben tener duración positiva"
        )

    max_overlap_ratio = np.zeros(
        len(result),
        dtype="float64",
    )

    failure_ids = pd.array(
        [pd.NA] * len(result),
        dtype="Int64",
    )

    failures = get_failure_intervals()

    for failure in failures.itertuples(index=False):
        overlap_start = result[
            "window_start_timestamp"
        ].clip(
            lower=failure.start_timestamp
        )

        overlap_end = result[
            "window_end_timestamp"
        ].clip(
            upper=failure.end_timestamp
        )

        overlap_seconds = (
            overlap_end
            - overlap_start
        ).dt.total_seconds().clip(lower=0)

        overlap_ratio = (
            overlap_seconds.to_numpy()
            / window_duration.to_numpy()
        )

        better_match = (
            overlap_ratio > max_overlap_ratio
        )

        max_overlap_ratio[
            better_match
        ] = overlap_ratio[
            better_match
        ]

        failure_ids[
            better_match
        ] = failure.failure_id

    result["failure_overlap_ratio"] = (
        max_overlap_ratio
    )

    result["failure_id"] = failure_ids

    result["is_failure"] = (
        result["failure_overlap_ratio"]
        >= overlap_threshold
    )

    return result
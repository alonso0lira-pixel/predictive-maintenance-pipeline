"""Orquestación del pipeline de procesamiento de MetroPT-3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from predictive_maintenance.storage import save_parquet
from predictive_maintenance.transformation import transform_dataset
from predictive_maintenance.validation import validate_dataset


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Ejecuta validación, transformación y persistencia del dataset."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset de entrada: {input_path}"
        )

    raw_df = pd.read_csv(input_path)

    validation_report = validate_dataset(raw_df)

    if not validation_report["is_valid"]:
        raise ValueError(
            "El dataset no supera las validaciones obligatorias"
        )

    transformed_df = transform_dataset(raw_df)

    saved_path = save_parquet(
        transformed_df,
        output_path,
    )

    status = (
        "completed_with_warnings"
        if validation_report["has_warnings"]
        else "completed"
    )

    return {
        "status": status,
        "input_path": input_path,
        "output_path": saved_path,
        "input_rows": raw_df.shape[0],
        "input_columns": raw_df.shape[1],
        "output_rows": transformed_df.shape[0],
        "output_columns": transformed_df.shape[1],
        "validation": validation_report,
    }
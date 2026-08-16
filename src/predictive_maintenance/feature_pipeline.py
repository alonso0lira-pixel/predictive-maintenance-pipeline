"""Pipeline para generar datasets de features por partición temporal."""

from __future__ import annotations

from pathlib import Path

from predictive_maintenance.features import (
    build_feature_dataset,
)
from predictive_maintenance.splitting import (
    temporal_split_by_segment,
)
from predictive_maintenance.storage import (
    load_parquet,
    save_parquet,
)
from predictive_maintenance.windowing import (
    generate_window_index,
)


def run_feature_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    window_size: int = 60,
    step_size: int = 30,
) -> dict[str, object]:
    """Genera y guarda features para train, validation y test."""

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset procesado: {input_path}"
        )

    df = load_parquet(input_path)

    splits = temporal_split_by_segment(
        df,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    split_reports: dict[str, dict[str, object]] = {}

    for split_name, split_df in splits.items():
        window_index = generate_window_index(
            split_df,
            window_size=window_size,
            step_size=step_size,
        )

        feature_df = build_feature_dataset(
            split_df,
            window_index,
        )

        output_path = (
            output_dir
            / f"{split_name}.parquet"
        )

        save_parquet(
            feature_df,
            output_path,
        )

        split_reports[split_name] = {
            "rows": len(feature_df),
            "columns": feature_df.shape[1],
            "segments": (
                feature_df["segment_id"].nunique()
                if not feature_df.empty
                else 0
            ),
            "output_path": output_path,
        }

    return {
        "status": "completed",
        "input_path": input_path,
        "output_dir": output_dir,
        "window_size": window_size,
        "step_size": step_size,
        "splits": split_reports,
    }
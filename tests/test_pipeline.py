from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance.pipeline import run_pipeline
from predictive_maintenance.storage import load_parquet
from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
)


def make_valid_raw_dataframe() -> pd.DataFrame:
    """Construye un pequeño dataset raw compatible con MetroPT-3."""

    data: dict[str, object] = {
        "Unnamed: 0": [0, 10, 20],
        "timestamp": [
            "2020-02-01 00:00:00",
            "2020-02-01 00:00:10",
            "2020-02-01 00:00:20",
        ],
    }

    for position, column in enumerate(ANALOG_COLUMNS):
        data[column] = [
            float(position),
            float(position + 1),
            float(position + 2),
        ]

    for column in BINARY_COLUMNS:
        data[column] = [0.0, 1.0, 0.0]

    return pd.DataFrame(data)


def test_run_pipeline_creates_processed_parquet(
    tmp_path: Path,
) -> None:
    """El pipeline completo genera el fichero procesado."""

    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.parquet"

    df = make_valid_raw_dataframe()
    df.to_csv(input_path, index=False)

    result = run_pipeline(
        input_path,
        output_path,
    )

    assert result["status"] == "completed"
    assert output_path.exists()

    processed_df = load_parquet(output_path)

    assert processed_df.shape == (3, 17)
    assert "Unnamed: 0" not in processed_df.columns

    for column in BINARY_COLUMNS:
        assert processed_df[column].dtype == "int8"


def test_run_pipeline_reports_temporal_warning(
    tmp_path: Path,
) -> None:
    """Un hueco temporal genera advertencia pero no detiene el pipeline."""

    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.parquet"

    df = make_valid_raw_dataframe()
    df.loc[2, "timestamp"] = "2020-02-01 00:00:24"
    df.to_csv(input_path, index=False)

    result = run_pipeline(
        input_path,
        output_path,
    )

    assert result["status"] == "completed_with_warnings"
    assert result["validation"]["has_warnings"] is True
    assert (
        result["validation"]["checks"]["temporal_gaps"]["total_gaps"]
        == 1
    )

    assert output_path.exists()


def test_run_pipeline_stops_if_validation_fails(
    tmp_path: Path,
) -> None:
    """El pipeline no procesa datos que no superan las validaciones."""

    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.parquet"

    df = make_valid_raw_dataframe()
    df.loc[1, "COMP"] = 2
    df.to_csv(input_path, index=False)

    with pytest.raises(
        ValueError,
        match="no supera las validaciones",
    ):
        run_pipeline(
            input_path,
            output_path,
        )

    assert not output_path.exists()


def test_run_pipeline_raises_if_input_does_not_exist(
    tmp_path: Path,
) -> None:
    """El pipeline informa cuando el fichero de entrada no existe."""

    input_path = tmp_path / "missing.csv"
    output_path = tmp_path / "processed.parquet"

    with pytest.raises(
        FileNotFoundError,
        match="dataset de entrada",
    ):
        run_pipeline(
            input_path,
            output_path,
        )
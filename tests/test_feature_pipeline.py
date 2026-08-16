import pandas as pd
import pytest

from predictive_maintenance.feature_pipeline import (
    run_feature_pipeline,
)
from predictive_maintenance.storage import (
    load_parquet,
    save_parquet,
)
from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
)


def make_processed_dataframe() -> pd.DataFrame:
    """Construye tres segmentos válidos para probar el pipeline."""

    rows = 9

    data: dict[str, object] = {
        "timestamp": pd.date_range(
            "2020-02-01",
            periods=rows,
            freq="10s",
        ),
        "segment_id": (
            [0] * 3
            + [1] * 3
            + [2] * 3
        ),
    }

    for column in ANALOG_COLUMNS:
        data[column] = [
            float(i)
            for i in range(rows)
        ]

    for column in BINARY_COLUMNS:
        data[column] = [
            i % 2
            for i in range(rows)
        ]

    return pd.DataFrame(data)


def test_run_feature_pipeline_creates_three_datasets(
    tmp_path,
) -> None:
    """Genera los archivos de train, validation y test."""

    input_path = tmp_path / "processed.parquet"
    output_dir = tmp_path / "features"

    save_parquet(
        make_processed_dataframe(),
        input_path,
    )

    report = run_feature_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        window_size=2,
        step_size=1,
    )

    assert report["status"] == "completed"

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        output_path = (
            output_dir
            / f"{split_name}.parquet"
        )

        assert output_path.exists()

        feature_df = load_parquet(
            output_path
        )

        assert len(feature_df) == 2
        assert feature_df.shape[1] == 49


def test_run_feature_pipeline_reports_configuration(
    tmp_path,
) -> None:
    """El informe conserva los parámetros de ventanado."""

    input_path = tmp_path / "processed.parquet"

    save_parquet(
        make_processed_dataframe(),
        input_path,
    )

    report = run_feature_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "features",
        window_size=2,
        step_size=1,
    )

    assert report["window_size"] == 2
    assert report["step_size"] == 1


def test_run_feature_pipeline_rejects_missing_input(
    tmp_path,
) -> None:
    """El pipeline falla claramente si no existe la entrada."""

    with pytest.raises(FileNotFoundError):
        run_feature_pipeline(
            input_path=tmp_path / "missing.parquet",
            output_dir=tmp_path / "features",
            window_size=2,
            step_size=1,
        )
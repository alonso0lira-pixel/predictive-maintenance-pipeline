import pandas as pd
import pytest

from predictive_maintenance.feature_pipeline import (
    run_cutoff_feature_pipeline,
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
def make_cutoff_processed_dataframe() -> pd.DataFrame:
    """Construye datos antes y después de un corte entre segmentos."""

    timestamps = list(
        pd.date_range(
            "2020-02-29 23:59:20",
            periods=4,
            freq="10s",
        )
    ) + list(
        pd.date_range(
            "2020-03-01 00:00:00",
            periods=4,
            freq="10s",
        )
    )

    rows = len(timestamps)

    data: dict[str, object] = {
        "timestamp": timestamps,
        "segment_id": (
            [0] * 4
            + [1] * 4
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


def test_run_cutoff_feature_pipeline_creates_datasets(
    tmp_path,
) -> None:
    """Genera train y evaluación a ambos lados del corte."""

    input_path = tmp_path / "processed.parquet"
    output_dir = tmp_path / "features"

    save_parquet(
        make_cutoff_processed_dataframe(),
        input_path,
    )

    report = run_cutoff_feature_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        cutoff_timestamp="2020-03-01 00:00:00",
        window_size=2,
        step_size=1,
    )

    assert report["status"] == "completed"

    train_path = (
        output_dir
        / "train_february.parquet"
    )

    evaluation_path = (
        output_dir
        / "evaluation_mar_sep.parquet"
    )

    assert train_path.exists()
    assert evaluation_path.exists()

    train = load_parquet(train_path)
    evaluation = load_parquet(evaluation_path)

    assert len(train) == 3
    assert len(evaluation) == 3

    assert train.shape[1] == 49
    assert evaluation.shape[1] == 49


def test_run_cutoff_feature_pipeline_preserves_configuration(
    tmp_path,
) -> None:
    """El informe conserva corte y configuración de ventanas."""

    input_path = tmp_path / "processed.parquet"

    save_parquet(
        make_cutoff_processed_dataframe(),
        input_path,
    )

    report = run_cutoff_feature_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "features",
        cutoff_timestamp="2020-03-01 00:00:00",
        window_size=2,
        step_size=1,
    )

    assert (
        report["cutoff_timestamp"]
        == "2020-03-01 00:00:00"
    )

    assert report["window_size"] == 2
    assert report["step_size"] == 1


def test_run_cutoff_feature_pipeline_rejects_missing_input(
    tmp_path,
) -> None:
    """Falla claramente si el Parquet de entrada no existe."""

    with pytest.raises(FileNotFoundError):
        run_cutoff_feature_pipeline(
            input_path=tmp_path / "missing.parquet",
            output_dir=tmp_path / "features",
            cutoff_timestamp="2020-03-01 00:00:00",
        )
from pathlib import Path

import pandas as pd
import pytest

from predictive_maintenance.storage import (
    load_parquet,
    save_parquet,
)


def make_processed_dataframe() -> pd.DataFrame:
    """Construye un DataFrame pequeño representativo del dataset procesado."""

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:10",
                ]
            ),
            "TP2": [1.0, 2.0],
            "COMP": pd.Series([0, 1], dtype="int8"),
        }
    )


def test_save_parquet_creates_file(tmp_path: Path) -> None:
    """Crea correctamente el fichero Parquet."""

    df = make_processed_dataframe()
    output_path = tmp_path / "processed.parquet"

    result_path = save_parquet(df, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.is_file()


def test_save_parquet_creates_parent_directories(
    tmp_path: Path,
) -> None:
    """Crea automáticamente las carpetas de destino."""

    df = make_processed_dataframe()
    output_path = (
        tmp_path
        / "data"
        / "processed"
        / "dataset.parquet"
    )

    save_parquet(df, output_path)

    assert output_path.exists()


def test_parquet_round_trip_preserves_dataframe(
    tmp_path: Path,
) -> None:
    """La escritura y lectura conservan valores y tipos."""

    original_df = make_processed_dataframe()
    output_path = tmp_path / "processed.parquet"

    save_parquet(original_df, output_path)
    loaded_df = load_parquet(output_path)

    pd.testing.assert_frame_equal(
        loaded_df,
        original_df,
    )


def test_load_parquet_raises_if_file_does_not_exist(
    tmp_path: Path,
) -> None:
    """Informa claramente cuando el fichero no existe."""

    missing_path = tmp_path / "missing.parquet"

    with pytest.raises(
        FileNotFoundError,
        match="No se encontró el fichero Parquet",
    ):
        load_parquet(missing_path)
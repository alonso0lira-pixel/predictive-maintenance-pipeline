import pandas as pd
import pytest

from predictive_maintenance.transformation import (
    remove_auxiliary_columns,
)


def test_remove_auxiliary_columns_removes_unnamed_column() -> None:
    """Elimina la columna auxiliar esperada."""

    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 10, 20],
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
                "2020-02-01 00:00:20",
            ],
            "TP2": [1.0, 2.0, 3.0],
        }
    )

    result = remove_auxiliary_columns(df)

    assert "Unnamed: 0" not in result.columns
    assert list(result.columns) == ["timestamp", "TP2"]


def test_remove_auxiliary_columns_does_not_modify_original() -> None:
    """La transformación conserva intacto el DataFrame recibido."""

    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 10],
            "TP2": [1.0, 2.0],
        }
    )

    result = remove_auxiliary_columns(df)

    assert "Unnamed: 0" in df.columns
    assert "Unnamed: 0" not in result.columns
    assert result is not df


def test_remove_auxiliary_columns_raises_if_column_is_missing() -> None:
    """Informa claramente cuando falta la columna esperada."""

    df = pd.DataFrame(
        {
            "timestamp": ["2020-02-01 00:00:00"],
            "TP2": [1.0],
        }
    )

    with pytest.raises(
        KeyError,
        match="columnas auxiliares esperadas",
    ):
        remove_auxiliary_columns(df)

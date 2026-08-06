import pandas as pd
import pytest

from predictive_maintenance.transformation import (
    normalize_timestamp,
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


def test_normalize_timestamp_converts_string_to_datetime() -> None:
    """Convierte correctamente una columna de texto a tipo datetime."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
            ],
            "TP2": [1.0, 2.0],
        }
    )

    result = normalize_timestamp(df)

    assert pd.api.types.is_datetime64_any_dtype(
        result["timestamp"]
    )
    assert result.loc[0, "timestamp"] == pd.Timestamp(
        "2020-02-01 00:00:00"
    )


def test_normalize_timestamp_does_not_modify_original() -> None:
    """La transformación mantiene intacto el DataFrame original."""

    df = pd.DataFrame(
        {
            "timestamp": ["2020-02-01 00:00:00"],
        }
    )

    original_df = df.copy(deep=True)

    result = normalize_timestamp(df)

    pd.testing.assert_frame_equal(df, original_df)

    assert not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        result["timestamp"]
    )
    assert result is not df


def test_normalize_timestamp_raises_if_column_is_missing() -> None:
    """Informa cuando no existe la columna temporal esperada."""

    df = pd.DataFrame(
        {
            "TP2": [1.0],
        }
    )

    with pytest.raises(
        KeyError,
        match="columna temporal",
    ):
        normalize_timestamp(df)


def test_normalize_timestamp_raises_for_invalid_value() -> None:
    """Rechaza valores que no pueden interpretarse como fechas."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "fecha_invalida",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="1 valores temporales no interpretables",
    ):
        normalize_timestamp(df)
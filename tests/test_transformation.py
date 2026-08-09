import pandas as pd
import pytest

from predictive_maintenance.transformation import (
    cast_binary_columns,
    normalize_timestamp,
    remove_auxiliary_columns,
    transform_dataset,
    add_temporal_segments
)

from predictive_maintenance.validation import BINARY_COLUMNS


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


def make_binary_dataframe() -> pd.DataFrame:
    """Construye un DataFrame pequeño con todas las señales binarias."""

    return pd.DataFrame(
        {
            column: [0.0, 1.0, 0.0]
            for column in BINARY_COLUMNS
        }
    )


def test_cast_binary_columns_converts_columns_to_int8() -> None:
    """Convierte todas las señales digitales al tipo int8."""

    df = make_binary_dataframe()

    result = cast_binary_columns(df)

    for column in BINARY_COLUMNS:
        assert result[column].dtype == "int8"
        assert result[column].tolist() == [0, 1, 0]


def test_cast_binary_columns_does_not_modify_original() -> None:
    """La transformación conserva intacto el DataFrame original."""

    df = make_binary_dataframe()
    original_df = df.copy(deep=True)

    result = cast_binary_columns(df)

    pd.testing.assert_frame_equal(df, original_df)

    for column in BINARY_COLUMNS:
        assert df[column].dtype != "int8"
        assert result[column].dtype == "int8"

    assert result is not df


def test_cast_binary_columns_raises_if_column_is_missing() -> None:
    """Informa cuando falta una señal binaria esperada."""

    df = make_binary_dataframe().drop(columns="COMP")

    with pytest.raises(
        KeyError,
        match="columnas binarias esperadas",
    ):
        cast_binary_columns(df)


@pytest.mark.parametrize(
    ("invalid_value", "expected_message"),
    [
        (None, "contiene valores nulos"),
        (2, "contiene valores no permitidos"),
    ],
)
def test_cast_binary_columns_rejects_invalid_data(
    invalid_value: object,
    expected_message: str,
) -> None:
    """Rechaza nulos y valores distintos de cero o uno."""

    df = make_binary_dataframe()
    df.loc[1, "COMP"] = invalid_value

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        cast_binary_columns(df)


def make_raw_dataframe() -> pd.DataFrame:
    """Construye un DataFrame pequeño con el esquema necesario para transformar."""

    data: dict[str, object] = {
        "Unnamed: 0": [0, 10, 20],
        "timestamp": [
            "2020-02-01 00:00:00",
            "2020-02-01 00:00:10",
            "2020-02-01 00:00:20",
        ],
        "TP2": [1.0, 2.0, 3.0],
    }

    for column in BINARY_COLUMNS:
        data[column] = [0.0, 1.0, 0.0]

    return pd.DataFrame(data)


def test_transform_dataset_applies_all_transformations() -> None:
    """Ejecuta correctamente todas las transformaciones del pipeline."""

    df = make_raw_dataframe()

    result = transform_dataset(df)

    assert "Unnamed: 0" not in result.columns

    assert pd.api.types.is_datetime64_any_dtype(
        result["timestamp"]
    )

    for column in BINARY_COLUMNS:
        assert result[column].dtype == "int8"

    assert result["TP2"].tolist() == [1.0, 2.0, 3.0]


def test_transform_dataset_does_not_modify_original() -> None:
    """La transformación integral conserva intactos los datos de entrada."""

    df = make_raw_dataframe()
    original_df = df.copy(deep=True)

    result = transform_dataset(df)

    pd.testing.assert_frame_equal(df, original_df)
    assert result is not df
    assert "Unnamed: 0" in df.columns


def test_transform_dataset_propagates_invalid_timestamp_error() -> None:
    """La transformación se detiene si encuentra una fecha no interpretable."""

    df = make_raw_dataframe()
    df.loc[1, "timestamp"] = "fecha_invalida"

    with pytest.raises(
        ValueError,
        match="valores temporales no interpretables",
    ):
        transform_dataset(df)


def test_add_temporal_segments_keeps_continuous_data_together() -> None:
    """Una serie continua pertenece al mismo segmento."""

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:10",
                    "2020-02-01 00:00:20",
                ]
            )
        }
    )

    result = add_temporal_segments(df)

    assert result["segment_id"].tolist() == [0, 0, 0]


def test_add_temporal_segments_creates_new_segment_after_gap() -> None:
    """Un intervalo superior a 13 segundos crea un nuevo segmento."""

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:10",
                    "2020-02-01 00:00:24",
                    "2020-02-01 00:00:34",
                ]
            )
        }
    )

    result = add_temporal_segments(df)

    assert result["segment_id"].tolist() == [0, 0, 1, 1]
    assert result["segment_id"].dtype == "int32"


def test_add_temporal_segments_detects_multiple_gaps() -> None:
    """Cada interrupción temporal inicia un segmento independiente."""

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:10",
                    "2020-02-01 01:00:00",
                    "2020-02-01 01:00:10",
                    "2020-02-02 00:00:00",
                ]
            )
        }
    )

    result = add_temporal_segments(df)

    assert result["segment_id"].tolist() == [
        0,
        0,
        1,
        1,
        2,
    ]


def test_add_temporal_segments_does_not_modify_original() -> None:
    """La segmentación conserva intacto el DataFrame original."""

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:20",
                ]
            )
        }
    )

    original_df = df.copy(deep=True)

    result = add_temporal_segments(df)

    pd.testing.assert_frame_equal(df, original_df)

    assert "segment_id" not in df.columns
    assert "segment_id" in result.columns


def test_add_temporal_segments_requires_datetime() -> None:
    """La función exige que timestamp ya esté normalizado."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
            ]
        }
    )

    with pytest.raises(
        TypeError,
        match="debe tener tipo datetime",
    ):
        add_temporal_segments(df)
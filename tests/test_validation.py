import pandas as pd

from predictive_maintenance.validation import (
    EXPECTED_COLUMNS,
    validate_chronological_order,
    validate_duplicate_rows,
    validate_duplicate_timestamps,
    validate_missing_values,
    validate_schema,
)


def test_validate_schema_accepts_expected_columns() -> None:
    """El esquema es válido cuando contiene exactamente las columnas esperadas."""

    df = pd.DataFrame(columns=sorted(EXPECTED_COLUMNS))

    result = validate_schema(df)

    assert result["is_valid"] is True
    assert result["missing_columns"] == []
    assert result["unexpected_columns"] == []


def test_validate_schema_detects_missing_column() -> None:
    """La validación detecta una columna obligatoria ausente."""

    columns = EXPECTED_COLUMNS - {"TP2"}
    df = pd.DataFrame(columns=sorted(columns))

    result = validate_schema(df)

    assert result["is_valid"] is False
    assert result["missing_columns"] == ["TP2"]
    assert result["unexpected_columns"] == []


def test_validate_schema_detects_unexpected_column() -> None:
    """La validación detecta una columna no definida en el esquema."""

    columns = EXPECTED_COLUMNS | {"columna_inventada"}
    df = pd.DataFrame(columns=sorted(columns))

    result = validate_schema(df)

    assert result["is_valid"] is False
    assert result["missing_columns"] == []
    assert result["unexpected_columns"] == ["columna_inventada"]


def test_validate_missing_values_accepts_complete_dataframe() -> None:
    """La validación se supera cuando no existen valores nulos."""

    df = pd.DataFrame(
        {
            "TP2": [1.0, 2.0, 3.0],
            "COMP": [1, 0, 1],
        }
    )

    result = validate_missing_values(df)

    assert result["is_valid"] is True
    assert result["total_missing_values"] == 0
    assert result["missing_by_column"] == {}


def test_validate_missing_values_detects_nulls() -> None:
    """La validación identifica los nulos y las columnas afectadas."""

    df = pd.DataFrame(
        {
            "TP2": [1.0, None, 3.0],
            "COMP": [1, 0, None],
        }
    )

    result = validate_missing_values(df)

    assert result["is_valid"] is False
    assert result["total_missing_values"] == 2
    assert result["missing_by_column"] == {
        "TP2": 1,
        "COMP": 1,
    }


def test_validate_missing_values_accepts_empty_dataframe() -> None:
    """Un DataFrame sin filas no contiene valores nulos."""

    df = pd.DataFrame(columns=["TP2", "COMP"])

    result = validate_missing_values(df)

    assert result["is_valid"] is True
    assert result["total_missing_values"] == 0
    assert result["missing_by_column"] == {}

def test_validate_duplicate_rows_accepts_unique_rows() -> None:
    """La validación se supera cuando todas las filas son diferentes."""

    df = pd.DataFrame(
        {
            "TP2": [1.0, 2.0, 3.0],
            "COMP": [1, 0, 1],
        }
    )

    result = validate_duplicate_rows(df)

    assert result["is_valid"] is True
    assert result["duplicate_rows"] == 0


def test_validate_duplicate_rows_detects_duplicate() -> None:
    """La validación detecta una repetición completa de una fila."""

    df = pd.DataFrame(
        {
            "TP2": [1.0, 2.0, 1.0],
            "COMP": [1, 0, 1],
        }
    )

    result = validate_duplicate_rows(df)

    assert result["is_valid"] is False
    assert result["duplicate_rows"] == 1


def test_validate_duplicate_rows_counts_repeated_occurrences() -> None:
    """La primera aparición es original y las posteriores son duplicadas."""

    df = pd.DataFrame(
        {
            "TP2": [1.0, 1.0, 1.0],
            "COMP": [1, 1, 1],
        }
    )

    result = validate_duplicate_rows(df)

    assert result["is_valid"] is False
    assert result["duplicate_rows"] == 2


def test_validate_duplicate_timestamps_accepts_unique_values() -> None:
    """La validación se supera cuando todos los timestamps son únicos."""

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:10",
                    "2020-02-01 00:00:20",
                ]
            ),
            "TP2": [1.0, 2.0, 3.0],
        }
    )

    result = validate_duplicate_timestamps(df)

    assert result["is_valid"] is True
    assert result["duplicate_timestamps"] == 0


def test_validate_duplicate_timestamps_detects_repeated_timestamp() -> None:
    """Detecta un timestamp repetido aunque los sensores sean diferentes."""

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:10",
                    "2020-02-01 00:00:10",
                ]
            ),
            "TP2": [1.0, 2.0, 9.0],
        }
    )

    result = validate_duplicate_timestamps(df)

    assert result["is_valid"] is False
    assert result["duplicate_timestamps"] == 1


def test_validate_duplicate_timestamps_counts_repetitions() -> None:
    """La primera aparición es original y las posteriores son duplicadas."""

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:00",
                    "2020-02-01 00:00:00",
                ]
            )
        }
    )

    result = validate_duplicate_timestamps(df)

    assert result["is_valid"] is False
    assert result["duplicate_timestamps"] == 2

def test_validate_chronological_order_accepts_sorted_timestamps() -> None:
    """La validación se supera cuando el tiempo avanza correctamente."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
                "2020-02-01 00:00:20",
            ]
        }
    )

    result = validate_chronological_order(df)

    assert result["is_valid"] is True
    assert result["invalid_timestamps"] == 0
    assert result["temporal_reversals"] == 0


def test_validate_chronological_order_detects_reversal() -> None:
    """La validación detecta un retroceso temporal."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:20",
                "2020-02-01 00:00:10",
            ]
        }
    )

    result = validate_chronological_order(df)

    assert result["is_valid"] is False
    assert result["invalid_timestamps"] == 0
    assert result["temporal_reversals"] == 1


def test_validate_chronological_order_detects_invalid_timestamp() -> None:
    """Un valor temporal no interpretable invalida la comprobación."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "fecha_invalida",
                "2020-02-01 00:00:20",
            ]
        }
    )

    result = validate_chronological_order(df)

    assert result["is_valid"] is False
    assert result["invalid_timestamps"] == 1
    assert result["temporal_reversals"] == 0


def test_validate_chronological_order_allows_equal_timestamps() -> None:
    """Un timestamp repetido no es un retroceso cronológico."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
            ]
        }
    )

    result = validate_chronological_order(df)

    assert result["is_valid"] is True
    assert result["invalid_timestamps"] == 0
    assert result["temporal_reversals"] == 0
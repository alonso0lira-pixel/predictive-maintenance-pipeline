import pandas as pd

from predictive_maintenance.validation import (
    EXPECTED_COLUMNS,
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
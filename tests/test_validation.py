import pandas as pd

from predictive_maintenance.validation import (
    EXPECTED_COLUMNS,
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
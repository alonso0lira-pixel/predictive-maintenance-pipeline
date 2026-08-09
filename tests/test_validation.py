import pandas as pd

from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
    EXPECTED_COLUMNS,
    validate_analog_values,
    validate_binary_values,
    validate_chronological_order,
    validate_duplicate_rows,
    validate_duplicate_timestamps,
    validate_missing_values,
    validate_schema,
    validate_temporal_gaps,
    validate_dataset
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

def test_validate_binary_values_accepts_zero_and_one() -> None:
    """Las señales digitales pueden contener únicamente cero y uno."""

    df = pd.DataFrame(
        {
            column: [0, 1, 0]
            for column in BINARY_COLUMNS
        }
    )

    result = validate_binary_values(df)

    assert result["is_valid"] is True
    assert result["missing_columns"] == []
    assert result["total_invalid_values"] == 0
    assert result["invalid_counts_by_column"] == {}
    assert result["invalid_values_by_column"] == {}


def test_validate_binary_values_detects_invalid_values() -> None:
    """La validación identifica valores fuera del dominio binario."""

    data = {
        column: [0, 1, 0]
        for column in BINARY_COLUMNS
    }

    data["COMP"] = [0, 2, 1]
    data["LPS"] = [-1, 1, 3]

    df = pd.DataFrame(data)

    result = validate_binary_values(df)

    assert result["is_valid"] is False
    assert result["missing_columns"] == []
    assert result["total_invalid_values"] == 3
    assert result["invalid_counts_by_column"] == {
        "COMP": 1,
        "LPS": 2,
    }
    assert result["invalid_values_by_column"] == {
        "COMP": [2],
        "LPS": [-1, 3],
    }


def test_validate_binary_values_ignores_nulls() -> None:
    """Los nulos se controlan mediante la validación específica de ausencias."""

    data = {
        column: [0, 1, 0]
        for column in BINARY_COLUMNS
    }

    data["COMP"] = [0, None, 1]

    df = pd.DataFrame(data)

    result = validate_binary_values(df)

    assert result["is_valid"] is True
    assert result["total_invalid_values"] == 0
    assert result["invalid_values_by_column"] == {}


def test_validate_binary_values_detects_missing_column() -> None:
    """La función informa si falta una de las señales binarias esperadas."""

    data = {
        column: [0, 1]
        for column in BINARY_COLUMNS
        if column != "COMP"
    }

    df = pd.DataFrame(data)

    result = validate_binary_values(df)

    assert result["is_valid"] is False
    assert result["missing_columns"] == ["COMP"]
    assert result["total_invalid_values"] == 0

def test_validate_analog_values_accepts_valid_columns() -> None:
    """Acepta columnas analógicas numéricas, finitas y variables."""

    df = pd.DataFrame(
        {
            column: [0.0, 1.0, 2.0]
            for column in ANALOG_COLUMNS
        }
    )

    result = validate_analog_values(df)

    assert result["is_valid"] is True
    assert result["missing_columns"] == []
    assert result["non_numeric_columns"] == []
    assert result["constant_columns"] == []
    assert result["total_infinite_values"] == 0
    assert result["infinite_counts_by_column"] == {}


def test_validate_analog_values_detects_missing_column() -> None:
    """Detecta una señal analógica ausente."""

    df = pd.DataFrame(
        {
            column: [0.0, 1.0]
            for column in ANALOG_COLUMNS
            if column != "TP2"
        }
    )

    result = validate_analog_values(df)

    assert result["is_valid"] is False
    assert result["missing_columns"] == ["TP2"]


def test_validate_analog_values_detects_non_numeric_column() -> None:
    """Detecta una señal analógica con tipo no numérico."""

    data = {
        column: [0.0, 1.0, 2.0]
        for column in ANALOG_COLUMNS
    }
    data["TP2"] = ["bajo", "medio", "alto"]

    df = pd.DataFrame(data)

    result = validate_analog_values(df)

    assert result["is_valid"] is False
    assert result["non_numeric_columns"] == ["TP2"]


def test_validate_analog_values_detects_infinite_values() -> None:
    """Detecta valores infinitos positivos y negativos."""

    data = {
        column: [0.0, 1.0, 2.0, 3.0]
        for column in ANALOG_COLUMNS
    }
    data["TP2"] = [0.0, 1.0, float("inf"), float("-inf")]

    df = pd.DataFrame(data)

    result = validate_analog_values(df)

    assert result["is_valid"] is False
    assert result["total_infinite_values"] == 2
    assert result["infinite_counts_by_column"] == {
        "TP2": 2,
    }


def test_validate_analog_values_detects_constant_column() -> None:
    """Detecta una señal analógica sin variabilidad."""

    data = {
        column: [0.0, 1.0, 2.0]
        for column in ANALOG_COLUMNS
    }
    data["H1"] = [5.0, 5.0, 5.0]

    df = pd.DataFrame(data)

    result = validate_analog_values(df)

    assert result["is_valid"] is False
    assert result["constant_columns"] == ["H1"]


def test_validate_analog_values_does_not_count_null_as_infinite() -> None:
    """Los nulos se controlan mediante su validación específica."""

    df = pd.DataFrame(
        {
            column: [0.0, None, 1.0]
            for column in ANALOG_COLUMNS
        }
    )

    result = validate_analog_values(df)

    assert result["is_valid"] is True
    assert result["total_infinite_values"] == 0
    assert result["constant_columns"] == []

def test_validate_temporal_gaps_accepts_normal_intervals() -> None:
    """Los intervalos de hasta 13 segundos no generan advertencia."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
                "2020-02-01 00:00:23",
            ]
        }
    )

    result = validate_temporal_gaps(df)

    assert result["is_valid"] is True
    assert result["has_warning"] is False
    assert result["invalid_timestamps"] == 0
    assert result["total_gaps"] == 0
    assert result["max_gap_seconds"] == 13.0


def test_validate_temporal_gaps_counts_gap_thresholds() -> None:
    """Clasifica los huecos según su duración."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:14",
                "2020-02-01 00:01:15",
                "2020-02-01 00:06:16",
                "2020-02-01 01:06:17",
            ]
        }
    )

    result = validate_temporal_gaps(df)

    assert result["is_valid"] is True
    assert result["has_warning"] is True
    assert result["total_gaps"] == 4
    assert result["gaps_over_1_minute"] == 3
    assert result["gaps_over_5_minutes"] == 2
    assert result["gaps_over_1_hour"] == 1
    assert result["max_gap_seconds"] == 3601.0


def test_validate_temporal_gaps_detects_invalid_timestamp() -> None:
    """Un timestamp no interpretable invalida el análisis temporal."""

    df = pd.DataFrame(
        {
            "timestamp": [
                "2020-02-01 00:00:00",
                "fecha_invalida",
                "2020-02-01 00:00:20",
            ]
        }
    )

    result = validate_temporal_gaps(df)

    assert result["is_valid"] is False
    assert result["invalid_timestamps"] == 1


def test_validate_temporal_gaps_handles_single_row() -> None:
    """Con una sola fila no existe ningún intervalo que analizar."""

    df = pd.DataFrame(
        {
            "timestamp": ["2020-02-01 00:00:00"]
        }
    )

    result = validate_temporal_gaps(df)

    assert result["is_valid"] is True
    assert result["has_warning"] is False
    assert result["total_gaps"] == 0
    assert result["max_gap_seconds"] is None

def make_valid_dataframe() -> pd.DataFrame:
    """Construye un DataFrame pequeño compatible con MetroPT-3."""

    data: dict[str, object] = {
        "Unnamed: 0": [0, 10, 20],
        "timestamp": pd.to_datetime(
            [
                "2020-02-01 00:00:00",
                "2020-02-01 00:00:10",
                "2020-02-01 00:00:20",
            ]
        ),
    }

    for position, column in enumerate(ANALOG_COLUMNS):
        data[column] = [
            float(position),
            float(position + 1),
            float(position + 2),
        ]

    for column in BINARY_COLUMNS:
        data[column] = [0, 1, 0]

    return pd.DataFrame(data)

def test_validate_dataset_accepts_valid_dataframe() -> None:
    """El informe global es válido cuando todos los controles se superan."""

    df = make_valid_dataframe()

    result = validate_dataset(df)

    assert result["is_valid"] is True
    assert result["has_warnings"] is False
    assert result["skipped_checks"] == []

    checks = result["checks"]

    assert checks["schema"]["is_valid"] is True
    assert checks["missing_values"]["is_valid"] is True
    assert checks["duplicate_rows"]["is_valid"] is True
    assert checks["duplicate_timestamps"]["is_valid"] is True
    assert checks["chronological_order"]["is_valid"] is True
    assert checks["binary_values"]["is_valid"] is True
    assert checks["analog_values"]["is_valid"] is True
    assert checks["temporal_gaps"]["is_valid"] is True


def test_validate_dataset_reports_temporal_warning() -> None:
    """Un hueco temporal genera advertencia sin invalidar el dataset."""

    df = make_valid_dataframe()

    df["timestamp"] = pd.to_datetime(
        [
            "2020-02-01 00:00:00",
            "2020-02-01 00:00:10",
            "2020-02-01 00:00:24",
        ]
    )

    result = validate_dataset(df)

    assert result["is_valid"] is True
    assert result["has_warnings"] is True
    assert result["checks"]["temporal_gaps"]["total_gaps"] == 1


def test_validate_dataset_becomes_invalid_for_binary_error() -> None:
    """Un valor digital incorrecto invalida el informe global."""

    df = make_valid_dataframe()
    df.loc[1, "COMP"] = 2

    result = validate_dataset(df)

    assert result["is_valid"] is False
    assert result["checks"]["binary_values"]["is_valid"] is False
    assert (
        result["checks"]["binary_values"]["total_invalid_values"]
        == 1
    )


def test_validate_dataset_skips_temporal_checks_without_timestamp() -> None:
    """Los controles temporales se omiten si falta la columna timestamp."""

    df = make_valid_dataframe().drop(columns="timestamp")

    result = validate_dataset(df)

    assert result["is_valid"] is False
    assert result["checks"]["schema"]["is_valid"] is False
    assert result["skipped_checks"] == [
        "duplicate_timestamps",
        "chronological_order",
        "temporal_gaps",
    ]
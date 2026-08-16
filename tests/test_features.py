import pandas as pd
import pytest

from predictive_maintenance.features import (
    aggregate_window_features,
    build_feature_dataset,
)
from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
)
from predictive_maintenance.windowing import (
    MODEL_FEATURE_COLUMNS,
)

def make_test_window() -> pd.DataFrame:
    """Construye una ventana pequeña con todas las señales requeridas."""

    data: dict[str, object] = {}

    for column in ANALOG_COLUMNS:
        data[column] = [1.0, 2.0, 3.0, 4.0]

    for column in BINARY_COLUMNS:
        data[column] = [0, 0, 1, 1]

    return pd.DataFrame(data)


def test_aggregate_window_features_returns_44_features() -> None:
    """La ventana genera el número esperado de features."""

    window = make_test_window()

    result = aggregate_window_features(window)

    assert len(result) == 44


def test_aggregate_window_features_calculates_analog_statistics() -> None:
    """Calcula correctamente los estadísticos analógicos."""

    window = make_test_window()

    result = aggregate_window_features(window)

    assert result["TP2_mean"] == 2.5
    assert result["TP2_min"] == 1.0
    assert result["TP2_max"] == 4.0

    assert result["TP2_std"] == pytest.approx(
        1.11803398875
    )


def test_aggregate_window_features_calculates_binary_features() -> None:
    """Calcula proporción activa y transiciones digitales."""

    window = make_test_window()

    result = aggregate_window_features(window)

    assert result["COMP_active_ratio"] == 0.5
    assert result["COMP_transitions"] == 1.0


def test_aggregate_window_features_rejects_missing_columns() -> None:
    """La generación falla si falta alguna señal requerida."""

    window = make_test_window().drop(columns="TP2")

    with pytest.raises(
        KeyError,
        match="Faltan columnas necesarias",
    ):
        aggregate_window_features(window)


def test_aggregate_window_features_rejects_empty_window() -> None:
    """No se generan features de una ventana sin registros."""

    window = make_test_window().iloc[0:0]

    with pytest.raises(
        ValueError,
        match="ventana vacía",
    ):
        aggregate_window_features(window)

def make_feature_source_dataframe() -> pd.DataFrame:
    """Construye un dataset pequeño para generar varias ventanas."""

    data: dict[str, object] = {}

    for column in ANALOG_COLUMNS:
        data[column] = [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]

    for column in BINARY_COLUMNS:
        data[column] = [
            0,
            0,
            1,
            1,
            0,
            0,
        ]

    return pd.DataFrame(data)


def test_build_feature_dataset_creates_one_row_per_window() -> None:
    """Genera exactamente una fila por cada ventana indexada."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [0, 0],
            "start_index": [0, 3],
            "end_index": [3, 6],
        }
    )

    result = build_feature_dataset(
        df,
        window_index,
    )

    assert len(result) == 2
    assert result.shape[1] == 47


def test_build_feature_dataset_preserves_window_metadata() -> None:
    """Mantiene los metadatos necesarios para rastrear cada ventana."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [5],
            "start_index": [1],
            "end_index": [4],
        }
    )

    result = build_feature_dataset(
        df,
        window_index,
    )

    assert result.loc[0, "segment_id"] == 5
    assert result.loc[0, "start_index"] == 1
    assert result.loc[0, "end_index"] == 4


def test_build_feature_dataset_calculates_features_for_each_window() -> None:
    """Los agregados corresponden a los registros de cada ventana."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [0, 0],
            "start_index": [0, 3],
            "end_index": [3, 6],
        }
    )

    result = build_feature_dataset(
        df,
        window_index,
    )

    assert result.loc[0, "TP2_mean"] == 2.0
    assert result.loc[1, "TP2_mean"] == 5.0

    assert result.loc[0, "COMP_active_ratio"] == pytest.approx(
        1 / 3
    )

    assert result.loc[1, "COMP_active_ratio"] == pytest.approx(
        1 / 3
    )


def test_build_feature_dataset_rejects_invalid_window_index() -> None:
    """El índice debe contener sus tres columnas obligatorias."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [0],
            "start_index": [0],
        }
    )

    with pytest.raises(
        KeyError,
        match="índice de ventanas",
    ):
        build_feature_dataset(
            df,
            window_index,
        )

def test_build_feature_dataset_matches_individual_aggregation() -> None:
    """La implementación vectorizada coincide con la agregación individual."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [0, 0],
            "start_index": [0, 3],
            "end_index": [3, 6],
        }
    )

    result = build_feature_dataset(
        df,
        window_index,
    )

    expected_first = aggregate_window_features(
        df.iloc[0:3]
    )

    expected_second = aggregate_window_features(
        df.iloc[3:6]
    )

    for feature_name in expected_first.index:
        assert result.loc[0, feature_name] == pytest.approx(
            expected_first[feature_name]
        )

        assert result.loc[1, feature_name] == pytest.approx(
            expected_second[feature_name]
        )
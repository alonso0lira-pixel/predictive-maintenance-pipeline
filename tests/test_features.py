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

    data["segment_id"] = [0] * 6    

    data["timestamp"] = pd.date_range(
        "2020-02-01",
        periods=6,
        freq="10s",
    )

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
    assert result.shape[1] == 49


def test_build_feature_dataset_preserves_window_metadata() -> None:
    """Mantiene los metadatos necesarios para rastrear cada ventana."""

    df = make_feature_source_dataframe()
    df["segment_id"] = 5    

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

def test_build_feature_dataset_preserves_temporal_metadata() -> None:
    """Cada ventana conserva correctamente sus límites temporales."""

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

    assert result.loc[
        0,
        "window_start_timestamp",
    ] == pd.Timestamp("2020-02-01 00:00:00")

    assert result.loc[
        0,
        "window_end_timestamp",
    ] == pd.Timestamp("2020-02-01 00:00:20")

    assert result.loc[
        1,
        "window_start_timestamp",
    ] == pd.Timestamp("2020-02-01 00:00:30")

    assert result.loc[
        1,
        "window_end_timestamp",
    ] == pd.Timestamp("2020-02-01 00:00:50")

def test_build_feature_dataset_requires_source_segment_id() -> None:
    """El dataset fuente debe contener el identificador de segmento."""

    df = make_feature_source_dataframe().drop(
        columns="segment_id"
    )

    window_index = pd.DataFrame(
        {
            "segment_id": [0],
            "start_index": [0],
            "end_index": [3],
        }
    )

    with pytest.raises(
        KeyError,
        match="columna segment_id",
    ):
        build_feature_dataset(
            df,
            window_index,
        )


def test_build_feature_dataset_rejects_mismatched_segment_id() -> None:
    """El segmento declarado debe coincidir con el dataset fuente."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [5],
            "start_index": [0],
            "end_index": [3],
        }
    )

    with pytest.raises(
        ValueError,
        match="no coincide",
    ):
        build_feature_dataset(
            df,
            window_index,
        )


def test_build_feature_dataset_rejects_cross_segment_window() -> None:
    """Una ventana no puede atravesar un límite entre segmentos."""

    df = make_feature_source_dataframe()
    df["segment_id"] = [
        0,
        0,
        0,
        1,
        1,
        1,
    ]

    window_index = pd.DataFrame(
        {
            "segment_id": [0],
            "start_index": [2],
            "end_index": [5],
        }
    )

    with pytest.raises(
        ValueError,
        match="atraviesan",
    ):
        build_feature_dataset(
            df,
            window_index,
        )


def test_build_feature_dataset_rejects_non_integer_bounds() -> None:
    """Los límites de las ventanas deben ser posiciones enteras."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [0],
            "start_index": [0.5],
            "end_index": [3],
        }
    )

    with pytest.raises(
        TypeError,
        match="debe contener enteros",
    ):
        build_feature_dataset(
            df,
            window_index,
        )


def test_build_feature_dataset_rejects_empty_window() -> None:
    """Una ventana debe tener longitud estrictamente positiva."""

    df = make_feature_source_dataframe()

    window_index = pd.DataFrame(
        {
            "segment_id": [0],
            "start_index": [2],
            "end_index": [2],
        }
    )

    with pytest.raises(
        ValueError,
        match="end_index > start_index",
    ):
        build_feature_dataset(
            df,
            window_index,
        )


def test_build_feature_dataset_preserves_window_index_order() -> None:
    """La salida conserva el orden recibido en window_index."""

    df = make_feature_source_dataframe()
    df["segment_id"] = [
        0,
        0,
        0,
        1,
        1,
        1,
    ]

    window_index = pd.DataFrame(
        {
            "segment_id": [
                0,
                1,
                0,
            ],
            "start_index": [
                0,
                3,
                1,
            ],
            "end_index": [
                2,
                5,
                3,
            ],
        }
    )

    result = build_feature_dataset(
        df,
        window_index,
    )

    assert list(result["start_index"]) == [
        0,
        3,
        1,
    ]

    assert list(result["segment_id"]) == [
        0,
        1,
        0,
    ]
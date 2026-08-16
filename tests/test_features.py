import pandas as pd
import pytest

from predictive_maintenance.features import (
    aggregate_window_features,
)
from predictive_maintenance.validation import (
    ANALOG_COLUMNS,
    BINARY_COLUMNS,
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
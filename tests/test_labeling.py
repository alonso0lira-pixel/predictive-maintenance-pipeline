import pandas as pd
import pytest

from predictive_maintenance.labeling import (
    label_failure_windows,
)


def test_window_without_failure_has_zero_overlap() -> None:
    """Una ventana fuera de los eventos no se etiqueta como fallo."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-03-01 00:00:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-03-01 00:10:00")
            ],
        }
    )

    result = label_failure_windows(windows)

    assert result.loc[0, "failure_overlap_ratio"] == 0.0
    assert pd.isna(result.loc[0, "failure_id"])
    assert not result.loc[0, "is_failure"]


def test_window_fully_inside_failure_has_full_overlap() -> None:
    """Una ventana completamente dentro del fallo tiene ratio 1."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-18 10:00:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-04-18 10:10:00")
            ],
        }
    )

    result = label_failure_windows(windows)

    assert result.loc[0, "failure_overlap_ratio"] == 1.0
    assert result.loc[0, "failure_id"] == 1
    assert result.loc[0, "is_failure"]


def test_half_overlap_is_classified_as_failure() -> None:
    """Un solapamiento del 50 % alcanza el umbral principal."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-17 23:55:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-04-18 00:05:00")
            ],
        }
    )

    result = label_failure_windows(
        windows,
        overlap_threshold=0.50,
    )

    assert result.loc[
        0,
        "failure_overlap_ratio",
    ] == pytest.approx(0.50)

    assert result.loc[0, "failure_id"] == 1
    assert result.loc[0, "is_failure"]


def test_overlap_below_threshold_is_not_failure() -> None:
    """Un solapamiento pequeño se conserva pero no se clasifica como fallo."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-17 23:52:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-04-18 00:02:00")
            ],
        }
    )

    result = label_failure_windows(
        windows,
        overlap_threshold=0.50,
    )

    assert result.loc[
        0,
        "failure_overlap_ratio",
    ] == pytest.approx(0.20)

    assert result.loc[0, "failure_id"] == 1
    assert not result.loc[0, "is_failure"]


def test_window_is_assigned_to_correct_failure() -> None:
    """Una ventana dentro del cuarto evento conserva su failure_id."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-07-15 15:00:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-07-15 15:10:00")
            ],
        }
    )

    result = label_failure_windows(windows)

    assert result.loc[0, "failure_overlap_ratio"] == 1.0
    assert result.loc[0, "failure_id"] == 4
    assert result.loc[0, "is_failure"]


def test_label_failure_windows_does_not_modify_original() -> None:
    """El etiquetado devuelve una copia sin alterar la entrada."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-18 10:00:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-04-18 10:10:00")
            ],
        }
    )

    original_columns = windows.columns.tolist()

    result = label_failure_windows(windows)

    assert windows.columns.tolist() == original_columns
    assert "failure_overlap_ratio" not in windows.columns
    assert "failure_id" not in windows.columns
    assert "is_failure" not in windows.columns

    assert "failure_overlap_ratio" in result.columns
    assert "failure_id" in result.columns
    assert "is_failure" in result.columns


def test_label_failure_windows_rejects_missing_columns() -> None:
    """Falla si faltan los límites temporales de las ventanas."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-18")
            ],
        }
    )

    with pytest.raises(
        KeyError,
        match="Faltan columnas temporales",
    ):
        label_failure_windows(windows)


@pytest.mark.parametrize(
    "overlap_threshold",
    [
        0,
        -0.1,
        1.1,
    ],
)
def test_label_failure_windows_rejects_invalid_threshold(
    overlap_threshold: float,
) -> None:
    """El umbral debe ser una proporción mayor que 0 y menor o igual que 1."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-18")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-04-18 00:10:00")
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="overlap_threshold",
    ):
        label_failure_windows(
            windows,
            overlap_threshold=overlap_threshold,
        )


def test_label_failure_windows_rejects_non_positive_duration() -> None:
    """Una ventana debe terminar después de comenzar."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": [
                pd.Timestamp("2020-04-18 10:00:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-04-18 10:00:00")
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="duración positiva",
    ):
        label_failure_windows(windows)


def test_label_failure_windows_handles_empty_dataframe() -> None:
    """Un DataFrame vacío conserva el esquema de etiquetas."""

    windows = pd.DataFrame(
        {
            "window_start_timestamp": pd.Series(
                dtype="datetime64[ns]"
            ),
            "window_end_timestamp": pd.Series(
                dtype="datetime64[ns]"
            ),
        }
    )

    result = label_failure_windows(windows)

    assert result.empty

    assert "failure_overlap_ratio" in result.columns
    assert "failure_id" in result.columns
    assert "is_failure" in result.columns

    assert str(result["failure_id"].dtype) == "Int64"
    assert result["is_failure"].dtype == bool
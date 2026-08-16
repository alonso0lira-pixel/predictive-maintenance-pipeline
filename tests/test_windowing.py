import pandas as pd
import pytest

from predictive_maintenance.windowing import (
    count_windows_by_segment,
)


def test_count_windows_by_segment_counts_windows() -> None:
    """Calcula correctamente las ventanas de cada segmento."""

    df = pd.DataFrame(
        {
            "segment_id": [
                0, 0, 0, 0, 0,
                1, 1, 1,
            ]
        }
    )

    result = count_windows_by_segment(
        df,
        window_size=3,
    )

    assert result["rows"].tolist() == [5, 3]
    assert result["windows"].tolist() == [3, 1]


def test_count_windows_by_segment_respects_step_size() -> None:
    """El desplazamiento controla la separación entre ventanas."""

    df = pd.DataFrame(
        {
            "segment_id": [0] * 10,
        }
    )

    result = count_windows_by_segment(
        df,
        window_size=4,
        step_size=2,
    )

    assert result.loc[0, "windows"] == 4


def test_count_windows_by_segment_returns_zero_for_short_segment() -> None:
    """Un segmento menor que la ventana no genera ejemplos."""

    df = pd.DataFrame(
        {
            "segment_id": [0, 0],
        }
    )

    result = count_windows_by_segment(
        df,
        window_size=3,
    )

    assert result.loc[0, "windows"] == 0


@pytest.mark.parametrize(
    ("window_size", "step_size"),
    [
        (0, 1),
        (-1, 1),
        (3, 0),
        (3, -1),
    ],
)
def test_count_windows_by_segment_rejects_invalid_parameters(
    window_size: int,
    step_size: int,
) -> None:
    """El tamaño y desplazamiento deben ser positivos."""

    df = pd.DataFrame(
        {
            "segment_id": [0, 0, 0],
        }
    )

    with pytest.raises(ValueError):
        count_windows_by_segment(
            df,
            window_size=window_size,
            step_size=step_size,
        )


def test_count_windows_by_segment_requires_segment_column() -> None:
    """La función exige que el dataset esté segmentado previamente."""

    df = pd.DataFrame(
        {
            "TP2": [1.0, 2.0],
        }
    )

    with pytest.raises(
        KeyError,
        match="columna de segmento",
    ):
        count_windows_by_segment(
            df,
            window_size=2,
        )
import pandas as pd
import pytest

from predictive_maintenance.windowing import (
    count_windows_by_segment,
    generate_window_index,
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


def test_generate_window_index_creates_expected_windows() -> None:
    """Genera correctamente los límites de las ventanas."""

    df = pd.DataFrame(
        {
            "segment_id": [0] * 5,
        }
    )

    result = generate_window_index(
        df,
        window_size=3,
        step_size=1,
    )

    assert result.to_dict("records") == [
        {
            "segment_id": 0,
            "start_index": 0,
            "end_index": 3,
        },
        {
            "segment_id": 0,
            "start_index": 1,
            "end_index": 4,
        },
        {
            "segment_id": 0,
            "start_index": 2,
            "end_index": 5,
        },
    ]


def test_generate_window_index_respects_step_size() -> None:
    """El desplazamiento controla el inicio de las ventanas."""

    df = pd.DataFrame(
        {
            "segment_id": [0] * 10,
        }
    )

    result = generate_window_index(
        df,
        window_size=4,
        step_size=2,
    )

    assert result["start_index"].tolist() == [
        0,
        2,
        4,
        6,
    ]

    assert result["end_index"].tolist() == [
        4,
        6,
        8,
        10,
    ]


def test_generate_window_index_never_crosses_segments() -> None:
    """Una ventana nunca contiene filas pertenecientes a dos segmentos."""

    df = pd.DataFrame(
        {
            "segment_id": [
                0, 0, 0,
                1, 1, 1,
            ]
        }
    )

    result = generate_window_index(
        df,
        window_size=2,
        step_size=1,
    )

    assert result.to_dict("records") == [
        {"segment_id": 0, "start_index": 0, "end_index": 2},
        {"segment_id": 0, "start_index": 1, "end_index": 3},
        {"segment_id": 1, "start_index": 3, "end_index": 5},
        {"segment_id": 1, "start_index": 4, "end_index": 6},
    ]


def test_generate_window_index_skips_short_segments() -> None:
    """Los segmentos menores que la ventana no generan índices."""

    df = pd.DataFrame(
        {
            "segment_id": [
                0, 0,
                1, 1, 1, 1,
            ]
        }
    )

    result = generate_window_index(
        df,
        window_size=3,
        step_size=1,
    )

    assert result["segment_id"].tolist() == [1, 1]
    assert result["start_index"].tolist() == [2, 3]


def test_generate_window_index_matches_window_count() -> None:
    """El número de índices coincide con el cálculo previo de ventanas."""

    df = pd.DataFrame(
        {
            "segment_id": [
                *([0] * 10),
                *([1] * 7),
            ]
        }
    )

    counts = count_windows_by_segment(
        df,
        window_size=4,
        step_size=2,
    )

    windows = generate_window_index(
        df,
        window_size=4,
        step_size=2,
    )

    assert len(windows) == int(counts["windows"].sum())
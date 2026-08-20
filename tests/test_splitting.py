import pandas as pd
import pytest

from predictive_maintenance.splitting import (
    temporal_split_by_cutoff,
    temporal_split_by_segment,
)


def make_segmented_dataframe() -> pd.DataFrame:
    """Construye cuatro segmentos cronológicos de distinto tamaño."""

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-02-01",
                periods=12,
                freq="10s",
            ),
            "segment_id": (
                [0] * 4
                + [1] * 3
                + [2] * 3
                + [3] * 2
            ),
            "value": range(12),
        }
    )


def test_temporal_split_preserves_all_rows() -> None:
    """Ningún registro se pierde durante la división."""

    df = make_segmented_dataframe()

    splits = temporal_split_by_segment(
        df,
        train_ratio=0.50,
        validation_ratio=0.25,
    )

    total_rows = sum(
        len(split)
        for split in splits.values()
    )

    assert total_rows == len(df)


def test_temporal_split_keeps_segments_separate() -> None:
    """Un segmento nunca aparece en más de un conjunto."""

    df = make_segmented_dataframe()

    splits = temporal_split_by_segment(
        df,
        train_ratio=0.50,
        validation_ratio=0.25,
    )

    train_segments = set(
        splits["train"]["segment_id"]
    )
    validation_segments = set(
        splits["validation"]["segment_id"]
    )
    test_segments = set(
        splits["test"]["segment_id"]
    )

    assert train_segments == {0, 1}
    assert validation_segments == {2}
    assert test_segments == {3}

    assert train_segments.isdisjoint(
        validation_segments
    )
    assert train_segments.isdisjoint(
        test_segments
    )
    assert validation_segments.isdisjoint(
        test_segments
    )


def test_temporal_split_preserves_chronology() -> None:
    """Train ocurre antes que validation y validation antes que test."""

    df = make_segmented_dataframe()

    splits = temporal_split_by_segment(
        df,
        train_ratio=0.50,
        validation_ratio=0.25,
    )

    assert (
        splits["train"]["timestamp"].max()
        < splits["validation"]["timestamp"].min()
    )

    assert (
        splits["validation"]["timestamp"].max()
        < splits["test"]["timestamp"].min()
    )


def test_temporal_split_rejects_invalid_ratios() -> None:
    """Las proporciones deben dejar espacio para los tres conjuntos."""

    df = make_segmented_dataframe()

    with pytest.raises(ValueError):
        temporal_split_by_segment(
            df,
            train_ratio=0,
            validation_ratio=0.20,
        )

    with pytest.raises(ValueError):
        temporal_split_by_segment(
            df,
            train_ratio=0.70,
            validation_ratio=0,
        )

    with pytest.raises(ValueError):
        temporal_split_by_segment(
            df,
            train_ratio=0.90,
            validation_ratio=0.20,
        )


def test_temporal_split_requires_three_segments() -> None:
    """Cada conjunto necesita al menos un segmento."""

    df = pd.DataFrame(
        {
            "segment_id": [0, 0, 1, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="al menos tres segmentos",
    ):
        temporal_split_by_segment(df)


def test_temporal_split_by_cutoff_preserves_rows() -> None:
    """El corte temporal conserva todos los registros."""

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-02-29 23:59:30",
                periods=6,
                freq="10s",
            ),
            "segment_id": [
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        }
    )

    cutoff = pd.Timestamp(
        "2020-03-01 00:00:00"
    )

    splits = temporal_split_by_cutoff(
        df,
        cutoff,
    )

    assert (
        len(splits["train"])
        + len(splits["evaluation"])
        == len(df)
    )


def test_temporal_split_by_cutoff_preserves_chronology() -> None:
    """Train ocurre completamente antes que evaluación."""

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-02-29 23:59:30",
                periods=6,
                freq="10s",
            ),
            "segment_id": [
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        }
    )

    splits = temporal_split_by_cutoff(
        df,
        "2020-03-01 00:00:00",
    )

    assert (
        splits["train"]["timestamp"].max()
        < splits["evaluation"]["timestamp"].min()
    )


def test_temporal_split_by_cutoff_keeps_segments_separate() -> None:
    """Ningún segmento puede aparecer en ambos conjuntos."""

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-02-29 23:59:30",
                periods=6,
                freq="10s",
            ),
            "segment_id": [
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        }
    )

    splits = temporal_split_by_cutoff(
        df,
        "2020-03-01 00:00:00",
    )

    train_segments = set(
        splits["train"]["segment_id"]
    )

    evaluation_segments = set(
        splits["evaluation"]["segment_id"]
    )

    assert train_segments.isdisjoint(
        evaluation_segments
    )


def test_temporal_split_by_cutoff_rejects_segment_cut() -> None:
    """El corte no puede atravesar un segmento continuo."""

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-02-29 23:59:40",
                periods=5,
                freq="10s",
            ),
            "segment_id": [
                0,
                0,
                0,
                0,
                0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="divide uno o más segmentos",
    ):
        temporal_split_by_cutoff(
            df,
            "2020-03-01 00:00:00",
        )


def test_temporal_split_by_cutoff_rejects_outside_range() -> None:
    """El corte debe dejar observaciones en ambos conjuntos."""

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-02-01",
                periods=6,
                freq="10s",
            ),
            "segment_id": [
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="tanto en train como en evaluación",
    ):
        temporal_split_by_cutoff(
            df,
            "2021-01-01",
        )
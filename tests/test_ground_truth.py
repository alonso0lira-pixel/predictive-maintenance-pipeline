import pandas as pd

from predictive_maintenance.ground_truth import (
    get_failure_intervals,
)


def test_get_failure_intervals_returns_four_failures() -> None:
    """MetroPT-3 contiene cuatro intervalos de fallo documentados."""

    failures = get_failure_intervals()

    assert len(failures) == 4
    assert failures["failure_id"].tolist() == [
        1,
        2,
        3,
        4,
    ]


def test_failure_intervals_are_datetime() -> None:
    """Los límites temporales se representan como datetime."""

    failures = get_failure_intervals()

    assert pd.api.types.is_datetime64_any_dtype(
        failures["start_timestamp"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        failures["end_timestamp"]
    )


def test_failure_intervals_have_valid_bounds() -> None:
    """Cada fallo termina después de comenzar."""

    failures = get_failure_intervals()

    assert (
        failures["end_timestamp"]
        > failures["start_timestamp"]
    ).all()


def test_failure_intervals_are_chronological() -> None:
    """Los eventos oficiales están ordenados cronológicamente."""

    failures = get_failure_intervals()

    assert failures[
        "start_timestamp"
    ].is_monotonic_increasing


def test_first_failure_matches_official_interval() -> None:
    """Comprueba uno de los intervalos contra el ground truth oficial."""

    failures = get_failure_intervals()

    first_failure = failures.iloc[0]

    assert first_failure["start_timestamp"] == pd.Timestamp(
        "2020-04-18 00:00:00"
    )

    assert first_failure["end_timestamp"] == pd.Timestamp(
        "2020-04-18 23:59:00"
    )
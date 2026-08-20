"""Ground truth de fallos documentados para MetroPT-3."""

from __future__ import annotations

import pandas as pd


METROPT3_FAILURES = (
    {
        "failure_id": 1,
        "start_timestamp": "2020-04-18 00:00:00",
        "end_timestamp": "2020-04-18 23:59:00",
        "failure_type": "Air leak",
        "severity": "High stress",
    },
    {
        "failure_id": 2,
        "start_timestamp": "2020-05-29 23:30:00",
        "end_timestamp": "2020-05-30 06:00:00",
        "failure_type": "Air leak",
        "severity": "High stress",
    },
    {
        "failure_id": 3,
        "start_timestamp": "2020-06-05 10:00:00",
        "end_timestamp": "2020-06-07 14:30:00",
        "failure_type": "Air leak",
        "severity": "High stress",
    },
    {
        "failure_id": 4,
        "start_timestamp": "2020-07-15 14:30:00",
        "end_timestamp": "2020-07-15 19:00:00",
        "failure_type": "Air leak",
        "severity": "High stress",
    },
)


def get_failure_intervals() -> pd.DataFrame:
    """Devuelve los intervalos de fallo documentados de MetroPT-3."""

    failures = pd.DataFrame(
        METROPT3_FAILURES
    )

    failures["start_timestamp"] = pd.to_datetime(
        failures["start_timestamp"]
    )

    failures["end_timestamp"] = pd.to_datetime(
        failures["end_timestamp"]
    )

    return failures
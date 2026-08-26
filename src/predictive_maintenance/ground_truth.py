"""Ground truth de fallos documentados para MetroPT-3.

Fuente:
    MetroPT-3 Data Description, sección "Failure Information".

Los timestamps reproducen los intervalos de fallo publicados en la
documentación del dataset. La fuente no especifica zona horaria, por lo
que se utilizan timestamps timezone-naive.

Los valores de ``failure_id`` son identificadores internos secuenciales
del proyecto. La documentación original repite el identificador "#1"
en su segunda fila, por lo que no se utiliza directamente esa numeración.
"""

from __future__ import annotations

import pandas as pd


_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


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
        failures["start_timestamp"],
        format=_TIMESTAMP_FORMAT,
        errors="raise",
    )

    failures["end_timestamp"] = pd.to_datetime(
        failures["end_timestamp"],
        format=_TIMESTAMP_FORMAT,
        errors="raise",
    )

    if failures["failure_id"].duplicated().any():
        raise ValueError(
            "Los failure_id deben ser únicos"
        )

    if (
        failures["end_timestamp"]
        <= failures["start_timestamp"]
    ).any():
        raise ValueError(
            "Todos los fallos deben tener "
            "end_timestamp > start_timestamp"
        )

    failures = (
        failures
        .sort_values("start_timestamp")
        .reset_index(drop=True)
    )

    previous_end = failures[
        "end_timestamp"
    ].shift()

    if (
        failures["start_timestamp"]
        < previous_end
    ).fillna(False).any():
        raise ValueError(
            "Los intervalos de fallo no deben solaparse"
        )

    return failures
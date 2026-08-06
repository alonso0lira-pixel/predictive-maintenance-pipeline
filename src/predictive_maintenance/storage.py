"""Lectura y escritura de datasets procesados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Guarda un DataFrame en Parquet sin incluir el índice."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        path,
        index=False,
    )

    return path


def load_parquet(
    input_path: str | Path,
) -> pd.DataFrame:
    """Carga un dataset almacenado en formato Parquet."""

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero Parquet: {path}"
        )

    return pd.read_parquet(path)
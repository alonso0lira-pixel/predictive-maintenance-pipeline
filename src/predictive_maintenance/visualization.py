"""Visualización de resultados del experimento de anomalías."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOCAL_PERIOD_ORDER = [
    "24-12h",
    "12-6h",
    "6-3h",
    "3-1h",
    "1-0h",
    "durante",
]


def plot_failure_roc_auc(
    failure_metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Genera una gráfica de ROC-AUC para cada fallo documentado."""

    required_columns = {
        "failure_id",
        "roc_auc",
    }

    missing_columns = sorted(
        required_columns - set(failure_metrics.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para la gráfica: "
            f"{missing_columns}"
        )

    if failure_metrics.empty:
        raise ValueError(
            "No se puede generar la gráfica con un DataFrame vacío"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = failure_metrics.sort_values("failure_id")

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(
        data["failure_id"].astype(str),
        data["roc_auc"],
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Fallo")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC por fallo documentado")

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


def plot_local_horizon_roc_auc(
    local_metrics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Representa el ROC-AUC por horizonte para cada fallo."""

    required_columns = {
        "failure_id",
        "period",
        "roc_auc",
    }

    missing_columns = sorted(
        required_columns - set(local_metrics.columns)
    )

    if missing_columns:
        raise KeyError(
            "Faltan columnas necesarias para la gráfica: "
            f"{missing_columns}"
        )

    if local_metrics.empty:
        raise ValueError(
            "No se puede generar la gráfica con un DataFrame vacío"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for failure_id in sorted(
        local_metrics["failure_id"].unique()
    ):
        subset = local_metrics[
            local_metrics["failure_id"] == failure_id
        ].set_index("period")

        ordered = subset.reindex(
            LOCAL_PERIOD_ORDER
        )

        ax.plot(
            LOCAL_PERIOD_ORDER,
            ordered["roc_auc"],
            marker="o",
            label=f"Fallo {failure_id}",
        )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Referencia AUC = 0,5",
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Horizonte temporal")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "Separación local por horizonte previo al fallo"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path
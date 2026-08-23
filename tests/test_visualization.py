import pandas as pd
import pytest

from predictive_maintenance.visualization import (
    plot_failure_roc_auc,
    plot_local_horizon_roc_auc,
)


def test_plot_failure_roc_auc_creates_file(
    tmp_path,
) -> None:
    metrics = pd.DataFrame(
        {
            "failure_id": [1, 2],
            "roc_auc": [0.90, 0.95],
        }
    )

    output_path = tmp_path / "failure_auc.png"

    result = plot_failure_roc_auc(
        metrics,
        output_path,
    )

    assert result.exists()
    assert result.stat().st_size > 0


def test_plot_local_horizon_roc_auc_creates_file(
    tmp_path,
) -> None:
    metrics = pd.DataFrame(
        {
            "failure_id": [
                1, 1, 1,
                2, 2, 2,
            ],
            "period": [
                "24-12h",
                "1-0h",
                "durante",
                "24-12h",
                "1-0h",
                "durante",
            ],
            "roc_auc": [
                0.60,
                0.85,
                0.95,
                0.40,
                0.55,
                0.99,
            ],
        }
    )

    output_path = tmp_path / "local_auc.png"

    result = plot_local_horizon_roc_auc(
        metrics,
        output_path,
    )

    assert result.exists()
    assert result.stat().st_size > 0


def test_plot_failure_roc_auc_rejects_missing_columns(
    tmp_path,
) -> None:
    metrics = pd.DataFrame(
        {
            "failure_id": [1],
        }
    )

    with pytest.raises(
        KeyError,
        match="Faltan columnas necesarias",
    ):
        plot_failure_roc_auc(
            metrics,
            tmp_path / "figure.png",
        )


def test_plot_local_horizon_roc_auc_rejects_empty_dataframe(
    tmp_path,
) -> None:
    metrics = pd.DataFrame(
        columns=[
            "failure_id",
            "period",
            "roc_auc",
        ]
    )

    with pytest.raises(
        ValueError,
        match="DataFrame vacío",
    ):
        plot_local_horizon_roc_auc(
            metrics,
            tmp_path / "figure.png",
        )
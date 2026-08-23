"""Orquestación reproducible del experimento de detección de anomalías."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import json

from predictive_maintenance.anomaly_detection import (
    score_anomalies,
    train_isolation_forest,
)
from predictive_maintenance.evaluation import (
    evaluate_global_scores,
    evaluate_local_horizons,
    evaluate_scores_by_failure,
)
from predictive_maintenance.ground_truth import get_failure_intervals
from predictive_maintenance.labeling import label_failure_windows
from predictive_maintenance.modeling import prepare_model_input

def save_experiment_results(
    report: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Guarda los resultados del experimento en archivos reutilizables."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    global_metrics_path = output_path / "global_metrics.json"
    failure_metrics_path = output_path / "failure_metrics.csv"
    local_metrics_path = output_path / "local_horizon_metrics.csv"

    global_metrics = report["global_metrics"]
    failure_metrics = report["failure_metrics"]
    local_metrics = report["local_horizon_metrics"]

    with global_metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            global_metrics,
            file,
            indent=2,
        )

    failure_metrics.to_csv(
        failure_metrics_path,
        index=False,
    )

    local_metrics.to_csv(
        local_metrics_path,
        index=False,
    )

    return {
        "global_metrics": global_metrics_path,
        "failure_metrics": failure_metrics_path,
        "local_horizon_metrics": local_metrics_path,
    }

def run_anomaly_experiment(
    train_features_path: str | Path,
    evaluation_features_path: str | Path,
    n_estimators: int = 100,
    random_state: int = 42,
    overlap_threshold: float = 0.50,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Ejecuta el experimento completo de detección de anomalías."""

    train = pd.read_parquet(train_features_path)
    evaluation = pd.read_parquet(evaluation_features_path)

    X_train, _ = prepare_model_input(train)
    X_evaluation, evaluation_metadata = prepare_model_input(
        evaluation
    )

    model = train_isolation_forest(
        X_train,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    scores = score_anomalies(
        model,
        X_evaluation,
    )

    labels = label_failure_windows(
        evaluation_metadata,
        overlap_threshold=overlap_threshold,
    )

    if len(scores) != len(labels):
        raise ValueError(
            "El número de scores no coincide con el número "
            "de ventanas etiquetadas"
        )

    results = pd.concat(
        [
            labels.reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )

    failure_intervals = get_failure_intervals()

    global_metrics = evaluate_global_scores(results)

    failure_metrics = evaluate_scores_by_failure(
        results
    )

    local_horizon_metrics = evaluate_local_horizons(
        results,
        failure_intervals,
    )

    report = {
        "global_metrics": global_metrics,
        "failure_metrics": failure_metrics,
        "local_horizon_metrics": local_horizon_metrics,
    }

    if output_dir is not None:
        report["output_files"] = save_experiment_results(
            report,
            output_dir,
        )

    return report
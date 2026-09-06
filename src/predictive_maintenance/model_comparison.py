"""Comparación de detectores con configuraciones y protocolo fijos."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from predictive_maintenance.anomaly_detection import (
    score_anomalies,
    train_dense_autoencoder,
    train_isolation_forest,
    train_local_outlier_factor,
    train_one_class_svm,
)
from predictive_maintenance.evaluation import (
    evaluate_global_scores,
    evaluate_local_horizons,
    evaluate_scores_by_failure,
)
from predictive_maintenance.ground_truth import get_failure_intervals
from predictive_maintenance.labeling import label_failure_windows
from predictive_maintenance.modeling import prepare_model_input


def save_model_comparison(
    report: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Guarda las tablas global y por fallo, sin índice adicional."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("model_comparison", "model_failure_metrics"):
        paths[name] = output_path / f"{name}.csv"
        report[name].to_csv(paths[name], index=False)
    return paths


def run_model_comparison(
    train_features_path: str | Path,
    evaluation_features_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Compara los cuatro baselines sobre todas las filas de los mismos archivos.

    El entrenamiento no recibe etiquetas. Los detectores ajustan su scaler
    solo con train. El etiquetado usa solapamiento >= 0.50 y la evaluación
    reutiliza las funciones del experimento original. Los tiempos incluyen
    escalado y son informativos, dependientes del hardware; no se selecciona
    ningún modelo ni se ajustan hiperparámetros a partir de resultados.

    Devuelve dos tablas exportables y los horizontes locales por modelo.
    Sin output_dir no escribe archivos.
    """

    train = pd.read_parquet(train_features_path)
    evaluation = pd.read_parquet(evaluation_features_path)
    X_train, _ = prepare_model_input(train)
    X_evaluation, metadata = prepare_model_input(evaluation)
    labels = label_failure_windows(metadata, overlap_threshold=0.50)
    failure_intervals = get_failure_intervals()

    trainers = (
        ("isolation_forest", train_isolation_forest,
         {"n_estimators": 100, "random_state": 42}),
        ("one_class_svm", train_one_class_svm,
         {"nu": 0.05, "gamma": "scale"}),
        ("local_outlier_factor", train_local_outlier_factor, {}),
        ("dense_autoencoder", train_dense_autoencoder, {}),
    )
    comparison_rows = []
    failure_tables = []
    local_tables = []
    for name, trainer, parameters in trainers:
        start = perf_counter()
        model = trainer(X_train, **parameters)
        training_seconds = perf_counter() - start
        start = perf_counter()
        scores = score_anomalies(model, X_evaluation)
        scoring_seconds = perf_counter() - start

        if len(scores) != len(labels):
            raise ValueError(
                "El número de scores no coincide con el número "
                "de ventanas etiquetadas"
            )
        results = pd.concat(
            [labels.reset_index(drop=True), scores.reset_index(drop=True)],
            axis=1,
        )
        comparison_rows.append({
            "model": name,
            **evaluate_global_scores(results),
            "training_seconds": training_seconds,
            "scoring_seconds": scoring_seconds,
        })
        failures = evaluate_scores_by_failure(results)
        failures.insert(0, "model", name)
        failure_tables.append(failures)
        local = evaluate_local_horizons(results, failure_intervals)
        local.insert(0, "model", name)
        local_tables.append(local)

    report = {
        "model_comparison": pd.DataFrame(comparison_rows),
        "model_failure_metrics": pd.concat(failure_tables, ignore_index=True),
        "model_local_horizon_metrics": pd.concat(local_tables, ignore_index=True),
    }
    if output_dir is not None:
        report["output_files"] = save_model_comparison(report, output_dir)
    return report

import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.anomaly_detection import (
    score_anomalies,
    train_isolation_forest,
)


def make_training_features() -> pd.DataFrame:
    """Construye datos normales con un punto claramente extremo."""

    return pd.DataFrame(
        {
            "feature_a": [
                0.0,
                0.1,
                -0.1,
                0.05,
                -0.05,
                0.02,
                10.0,
            ],
            "feature_b": [
                0.0,
                0.1,
                -0.1,
                -0.05,
                0.05,
                0.02,
                10.0,
            ],
        }
    )


def test_train_isolation_forest_fits_model() -> None:
    """El modelo queda ajustado con el número correcto de features."""

    features = make_training_features()

    model = train_isolation_forest(
        features,
        n_estimators=20,
    )

    assert model.n_features_in_ == 2
    assert len(model.estimators_) == 20


def test_isolation_forest_is_reproducible() -> None:
    """La misma semilla produce las mismas puntuaciones."""

    features = make_training_features()

    first_model = train_isolation_forest(
        features,
        n_estimators=20,
        random_state=42,
    )

    second_model = train_isolation_forest(
        features,
        n_estimators=20,
        random_state=42,
    )

    first_scores = score_anomalies(
        first_model,
        features,
    )

    second_scores = score_anomalies(
        second_model,
        features,
    )

    np.testing.assert_allclose(
        first_scores["anomaly_score"],
        second_scores["anomaly_score"],
    )


def test_score_anomalies_returns_one_score_per_row() -> None:
    """Cada ventana recibe una puntuación y una clasificación."""

    features = make_training_features()

    model = train_isolation_forest(
        features,
        n_estimators=20,
    )

    result = score_anomalies(
        model,
        features,
    )

    assert len(result) == len(features)

    assert result.columns.tolist() == [
        "anomaly_score",
        "is_anomaly",
    ]

    assert result["anomaly_score"].notna().all()
    assert result["is_anomaly"].dtype == bool


def test_extreme_point_has_higher_anomaly_score() -> None:
    """El punto extremo recibe mayor score que la región normal."""

    features = make_training_features()

    model = train_isolation_forest(
        features,
        n_estimators=100,
    )

    result = score_anomalies(
        model,
        features,
    )

    extreme_score = result.iloc[-1][
        "anomaly_score"
    ]

    normal_scores = result.iloc[:-1][
        "anomaly_score"
    ]

    assert extreme_score > normal_scores.mean()


@pytest.mark.parametrize(
    "invalid_features",
    [
        pd.DataFrame(),
        pd.DataFrame(
            {"x": [1.0, np.nan]}
        ),
        pd.DataFrame(
            {"x": [1.0, np.inf]}
        ),
    ],
)
def test_train_isolation_forest_rejects_invalid_values(
    invalid_features,
) -> None:
    """El modelo rechaza matrices vacías o no finitas."""

    with pytest.raises(ValueError):
        train_isolation_forest(
            invalid_features
        )


def test_train_isolation_forest_rejects_non_numeric_features() -> None:
    """El modelo no acepta metadatos textuales como features."""

    features = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
            "metadata": ["a", "b"],
        }
    )

    with pytest.raises(
        TypeError,
        match="numéricas",
    ):
        train_isolation_forest(features)


def test_train_isolation_forest_rejects_invalid_estimators() -> None:
    """El bosque debe contener al menos un estimador."""

    features = make_training_features()

    with pytest.raises(
        ValueError,
        match="n_estimators",
    ):
        train_isolation_forest(
            features,
            n_estimators=0,
        )
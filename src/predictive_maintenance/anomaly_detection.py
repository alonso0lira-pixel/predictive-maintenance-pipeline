"""Detección no supervisada de anomalías."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted


def _validate_features(
    features: pd.DataFrame,
) -> None:
    """Valida la matriz de entrada utilizada por el modelo."""

    if features.empty:
        raise ValueError(
            "La matriz de features no puede estar vacía"
        )

    non_numeric_columns = (
        features.select_dtypes(
            exclude="number"
        ).columns.tolist()
    )

    if non_numeric_columns:
        raise TypeError(
            "Todas las features deben ser numéricas: "
            f"{non_numeric_columns}"
        )

    values = features.to_numpy(
        dtype="float64"
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Las features contienen valores no finitos"
        )


def train_isolation_forest(
    features: pd.DataFrame,
    n_estimators: int = 100,
    random_state: int = 42,
) -> IsolationForest:
    """Entrena un Isolation Forest sobre las features de train."""

    _validate_features(features)

    if n_estimators <= 0:
        raise ValueError(
            "n_estimators debe ser mayor que cero"
        )

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(features)

    return model


def train_one_class_svm(
    features: pd.DataFrame,
    nu: float = 0.05,
    gamma: str | float = "scale",
) -> Pipeline:
    """Entrena StandardScaler y OneClassSVM exclusivamente con train.

    No recibe etiquetas ni datos de evaluación. El pipeline conserva el
    scaler ajustado y lo reutiliza al puntuar nuevas ventanas.
    """

    _validate_features(features)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)),
        ]
    )
    model.fit(features)
    return model


def train_local_outlier_factor(
    features: pd.DataFrame,
) -> Pipeline:
    """Entrena StandardScaler y LOF exclusivamente con train, sin etiquetas.

    Usa novelty=True para puntuar datos nuevos mediante score_anomalies.
    El scaler ajustado se conserva para transformar evaluación.
    """

    _validate_features(features)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lof",
                LocalOutlierFactor(
                    n_neighbors=20,
                    contamination="auto",
                    novelty=True,
                ),
            ),
        ]
    )
    model.fit(features)
    return model


class DenseAutoencoderDetector(BaseEstimator):
    """Autoencoder denso con escalado y umbral aprendidos solo con train.

    La red aprende X_scaled → X_scaled sin etiquetas de fallo. Sus scores
    son el MSE negativo, compatible con la inversión de score_anomalies.
    """

    def fit(self, features: pd.DataFrame) -> DenseAutoencoderDetector:
        """Ajusta scaler, red y percentil 95 del MSE de entrenamiento."""

        _validate_features(features)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        network = MLPRegressor(
            hidden_layer_sizes=(32, 16, 8, 16, 32),
            activation="relu",
            solver="adam",
            loss="squared_error",
            learning_rate_init=0.001,
            batch_size=128,
            max_iter=100,
            alpha=0.0001,
            random_state=42,
            early_stopping=False,
            shuffle=True,
        )
        network.fit(scaled, scaled)
        errors = self._reconstruction_errors(network, scaled)

        self.scaler_ = scaler
        self.network_ = network
        self.threshold_ = float(np.percentile(errors, 95))
        self.n_features_in_ = features.shape[1]
        self.feature_names_in_ = features.columns.to_numpy(copy=True)
        return self

    @staticmethod
    def _reconstruction_errors(
        network: MLPRegressor, scaled: np.ndarray,
    ) -> np.ndarray:
        # MLPRegressor devuelve un vector para una sola feature de salida.
        reconstruction = network.predict(scaled).reshape(scaled.shape)
        return np.mean((scaled - reconstruction) ** 2, axis=1)

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        """Devuelve el MSE negativo sin ajustar ningún componente."""

        _validate_features(features)
        check_is_fitted(self, ["scaler_", "network_", "threshold_"])
        if not features.columns.equals(pd.Index(self.feature_names_in_)):
            raise ValueError("El esquema de features no coincide con entrenamiento")
        scaled = self.scaler_.transform(features)
        return -self._reconstruction_errors(self.network_, scaled)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Marca -1 solo cuando el error supera el umbral de train."""

        errors = -self.score_samples(features)
        return np.where(errors > self.threshold_, -1, 1)


def train_dense_autoencoder(features: pd.DataFrame) -> DenseAutoencoderDetector:
    """Entrena el autoencoder baseline sin etiquetas ni evaluación."""

    return DenseAutoencoderDetector().fit(features)


def score_anomalies(
    model: IsolationForest | Pipeline | DenseAutoencoderDetector,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula puntuaciones y clasificación de anomalía."""

    _validate_features(features)

    anomaly_score = -model.score_samples(
        features
    )

    prediction = model.predict(
        features
    )

    return pd.DataFrame(
        {
            "anomaly_score": anomaly_score,
            "is_anomaly": prediction == -1,
        },
        index=features.index,
    )

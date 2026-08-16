import pandas as pd
import pytest

from predictive_maintenance.modeling import (
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_COLUMNS,
    prepare_model_input,
)


def make_feature_dataframe() -> pd.DataFrame:
    """Construye una fila compatible con el dataset de features."""

    data: dict[str, object] = {
        column: [float(i)]
        for i, column in enumerate(
            MODEL_INPUT_COLUMNS
        )
    }

    data.update(
        {
            "segment_id": [3],
            "start_index": [100],
            "end_index": [160],
            "window_start_timestamp": [
                pd.Timestamp("2020-02-01 00:00:00")
            ],
            "window_end_timestamp": [
                pd.Timestamp("2020-02-01 00:09:50")
            ],
        }
    )

    return pd.DataFrame(data)


def test_model_input_contains_44_features() -> None:
    """El modelo recibe exactamente las 44 features."""

    feature_df = make_feature_dataframe()

    features, _ = prepare_model_input(
        feature_df
    )

    assert features.shape == (1, 44)
    assert features.columns.tolist() == MODEL_INPUT_COLUMNS


def test_model_metadata_contains_five_columns() -> None:
    """Los cinco metadatos quedan separados del modelo."""

    feature_df = make_feature_dataframe()

    _, metadata = prepare_model_input(
        feature_df
    )

    assert metadata.shape == (1, 5)
    assert (
        metadata.columns.tolist()
        == MODEL_METADATA_COLUMNS
    )


def test_model_input_excludes_metadata() -> None:
    """Ningún metadato forma parte de las variables del modelo."""

    feature_df = make_feature_dataframe()

    features, _ = prepare_model_input(
        feature_df
    )

    assert set(features.columns).isdisjoint(
        MODEL_METADATA_COLUMNS
    )


def test_prepare_model_input_rejects_missing_feature() -> None:
    """Falla si falta alguna feature requerida."""

    feature_df = (
        make_feature_dataframe()
        .drop(columns="TP2_mean")
    )

    with pytest.raises(
        KeyError,
        match="Faltan columnas necesarias",
    ):
        prepare_model_input(feature_df)
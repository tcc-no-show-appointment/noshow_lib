"""Testes para a feature `cluster_patient` (K-Means) introduzida em 0.4.0.

Cobre:
- fit_patient_cluster() treina o modelo e retorna artefato válido
- save/load round-trip do artefato
- apply_patient_cluster() atribui o cluster correto a pacientes conhecidos
- apply_patient_cluster() retorna -1 para pacientes novos
- apply_patient_cluster() retorna -1 quando artefato é vazio
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from noshow_lib.feature_engineering import (
    fit_patient_cluster,
    apply_patient_cluster,
    save_cluster_artifact,
    load_cluster_artifact,
    CLUSTER_FEATURES,
    CLUSTER_ARTIFACT_NAME,
)


def _make_training_df(n_patients: int = 30, seed: int = 42) -> pd.DataFrame:
    """Gera um DataFrame sintético com as features que o K-Means consome."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        # Cada paciente tem 3 consultas para gerar histórico suficiente
        for visit in range(3):
            rows.append({
                "patient_id": f"P{pid:03d}",
                "appointment_at": pd.Timestamp("2024-01-01") + pd.Timedelta(days=pid * 10 + visit),
                "previous_appointments_count": visit,
                "past_no_shows": int(rng.integers(0, visit + 1)),
                "no_show_rate_patient_smoothed": float(rng.uniform(0.0, 0.8)),
                "cancellation_rate": float(rng.uniform(0.0, 0.5)),
                "days_since_last_visit": -1.0 if visit == 0 else float(rng.integers(10, 200)),
                "waiting_days": int(rng.integers(0, 60)),
                "patient_age": int(rng.integers(18, 80)),
            })
    return pd.DataFrame(rows)


def test_fit_patient_cluster_basic():
    df = _make_training_df()
    artifact = fit_patient_cluster(df, k_range=(2, 4), random_state=0)

    assert artifact, "Artefato vazio"
    assert "scaler" in artifact
    assert "kmeans" in artifact
    assert "cluster_map" in artifact
    assert 2 <= artifact["best_k"] <= 4
    assert -1.0 <= artifact["best_score"] <= 1.0
    assert set(artifact["features"]).issubset(set(CLUSTER_FEATURES))
    # Todo paciente do treino deve estar no map
    assert len(artifact["cluster_map"]) == df["patient_id"].nunique()


def test_fit_patient_cluster_insufficient_data():
    """Quando há poucos pacientes (< max k), retorna dict vazio."""
    df = _make_training_df(n_patients=3)
    artifact = fit_patient_cluster(df, k_range=(2, 10))
    assert artifact == {}


def test_apply_patient_cluster_known_patients():
    df = _make_training_df(n_patients=20)
    artifact = fit_patient_cluster(df, k_range=(2, 4), random_state=0)

    out = apply_patient_cluster(df, artifact)
    assert "cluster_patient" in out.columns
    # Todos os pacientes do treino devem ter cluster != -1
    assert (out["cluster_patient"] != -1).all()
    # Valores devem estar em [0, best_k)
    assert out["cluster_patient"].min() >= 0
    assert out["cluster_patient"].max() < artifact["best_k"]


def test_apply_patient_cluster_unknown_patient():
    df = _make_training_df(n_patients=20)
    artifact = fit_patient_cluster(df, k_range=(2, 4), random_state=0)

    new_df = pd.DataFrame([{
        "patient_id": "P_NEW",
        "previous_appointments_count": 0,
        "past_no_shows": 0,
        "no_show_rate_patient_smoothed": 0.0,
        "cancellation_rate": 0.0,
        "days_since_last_visit": -1.0,
        "waiting_days": 5,
        "patient_age": 30,
    }])
    out = apply_patient_cluster(new_df, artifact)
    assert out["cluster_patient"].iloc[0] == -1


def test_apply_patient_cluster_empty_artifact():
    df = _make_training_df(n_patients=5)
    out = apply_patient_cluster(df, {})
    assert (out["cluster_patient"] == -1).all()


def test_apply_patient_cluster_missing_patient_id():
    """Sem patient_id na entrada, cluster vira -1 sem erro."""
    df = pd.DataFrame({"waiting_days": [1, 2, 3]})
    artifact = {"cluster_map": {"P1": 0}}  # artefato fake
    out = apply_patient_cluster(df, artifact)
    assert (out["cluster_patient"] == -1).all()


def test_save_load_round_trip(tmp_path: Path):
    df = _make_training_df(n_patients=20)
    artifact = fit_patient_cluster(df, k_range=(2, 4), random_state=0)

    save_path = save_cluster_artifact(artifact, tmp_path)
    assert save_path.name == CLUSTER_ARTIFACT_NAME
    assert save_path.exists()

    loaded = load_cluster_artifact(tmp_path)
    assert loaded is not None
    assert loaded["best_k"] == artifact["best_k"]
    assert len(loaded["cluster_map"]) == len(artifact["cluster_map"])

    # Aplicar o artefato carregado deve dar o mesmo resultado
    out_orig = apply_patient_cluster(df, artifact)
    out_loaded = apply_patient_cluster(df, loaded)
    pd.testing.assert_series_equal(
        out_orig["cluster_patient"], out_loaded["cluster_patient"], check_names=False
    )


def test_load_cluster_artifact_missing(tmp_path: Path):
    """Quando o arquivo não existe, load retorna None (sem erro)."""
    loaded = load_cluster_artifact(tmp_path)
    assert loaded is None

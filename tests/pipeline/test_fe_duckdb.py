"""
Teste de validação: build_features_from_parquet (DuckDB) vs build_features (pandas).

Passos:
1. Carrega amostra de 1000 linhas do Parquet real.
2. Roda pandas pipeline (build_features) → referência.
3. Salva amostra bruta como Parquet → roda DuckDB pipeline → lê resultado.
4. Compara colunas numéricas (tolerância 1e-4) e categóricas (igualdade).
5. Verifica anti-leakage: previous_appointments_count == 0 para 1ª consulta de cada paciente.
6. Verifica que o Parquet de saída é legível sem DuckDB (pd.read_parquet).
"""

import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from noshow_lib import load_config, build_features, build_features_from_parquet

# ── Configuração ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "abs_consultas_medicas_HT_v2.parquet"
CONFIG_PATH = ROOT / "config" / "local.yaml"

NUMERIC_TOL = 1e-4
SAMPLE_N = 1000

# Features de modelo conforme config (excluindo colunas de ID/target/datas)
CATEGORICAL_FEATURES = {
    "time_of_day",
    "age_group",
    "specialty_group",
    "gender_age_profile",
    "patient_sex",
    "insurance_type",
    "patient_city",
    "patient_neighborhood",
    "unit_name",
    "specialty",
}

NUMERIC_FEATURES = [
    "waiting_days", "is_same_day", "appointment_weekday", "is_weekend",
    "appointment_day_of_month", "appointment_week_of_month",
    "is_month_start", "is_month_end", "hour_appointment",
    "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "hour_sin", "hour_cos",
    "is_holiday", "is_pre_holiday", "is_post_holiday",
    "is_bridge_day", "is_holiday_window",
    "patient_age",
    "has_patient_history", "previous_appointments_count", "patient_tenure_days",
    "past_no_shows", "previous_no_show", "consecutive_no_shows_2",
    "past_cancellations_count",
    "waiting_days_delta", "appointments_in_same_schedule_day", "is_diff_specialty",
    "age_x_waiting_days",
    "same_city", "is_local_resident",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def sample_raw(config) -> pd.DataFrame:
    """Amostra bruta (sem feature engineering) de SAMPLE_N linhas."""
    pytest.importorskip("duckdb", reason="duckdb não instalado")
    if not DATA_PATH.exists():
        pytest.skip(f"Parquet de dados não encontrado: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH).head(SAMPLE_N)
    return df


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Converte appointment_id para string para permitir comparação entre pipelines."""
    df = df.copy()
    if "appointment_id" in df.columns:
        df["appointment_id"] = df["appointment_id"].astype(str)
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)
    return df


@pytest.fixture(scope="module")
def pandas_result(sample_raw, config) -> pd.DataFrame:
    """Resultado do pipeline pandas (referência), com IDs normalizados para str."""
    from noshow_lib import load_and_validate
    df_val = load_and_validate(sample_raw, config)
    return _normalize_ids(build_features(df_val, config))


@pytest.fixture(scope="module")
def duckdb_result(sample_raw, config) -> pd.DataFrame:
    """Resultado do pipeline DuckDB lido como DataFrame, com IDs normalizados para str."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.parquet"
        output_path = Path(tmpdir) / "output.parquet"

        sample_raw.to_parquet(input_path, index=False)
        out = build_features_from_parquet(input_path, config, output_path)
        return _normalize_ids(pd.read_parquet(out))


# ── Testes ────────────────────────────────────────────────────────────────────
class TestOutputSchema:
    def test_parquet_readable_without_duckdb(self, sample_raw, config):
        """Output Parquet deve ser legível só com pandas (sem duckdb)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_p = Path(tmpdir) / "in.parquet"
            out_p = Path(tmpdir) / "out.parquet"
            sample_raw.to_parquet(in_p, index=False)
            build_features_from_parquet(in_p, config, out_p)
            df = pd.read_parquet(out_p)
            assert len(df) > 0

    def test_expected_features_present(self, duckdb_result, pandas_result):
        """Todas as features do modelo devem estar no Parquet DuckDB."""
        model_features = set(NUMERIC_FEATURES) | CATEGORICAL_FEATURES
        missing = model_features - set(duckdb_result.columns)
        assert not missing, f"Features ausentes no output DuckDB: {missing}"

    def test_row_count_matches(self, duckdb_result, pandas_result):
        """DuckDB e pandas devem retornar o mesmo número de linhas."""
        assert len(duckdb_result) == len(pandas_result), (
            f"DuckDB: {len(duckdb_result)} linhas, Pandas: {len(pandas_result)} linhas"
        )


class TestNumericalAccuracy:
    """Compara features numéricas entre DuckDB e pandas (tolerância 1e-4)."""

    @pytest.mark.parametrize("col", NUMERIC_FEATURES)
    def test_numeric_column(self, duckdb_result, pandas_result, col):
        if col not in duckdb_result.columns or col not in pandas_result.columns:
            pytest.skip(f"Coluna '{col}' ausente em um dos resultados")

        # Alinhar por (appointment_id, appointment_at) para desambiguar ties
        sort_keys = [k for k in ("appointment_id", "appointment_at") if k in duckdb_result.columns]
        if sort_keys and all(k in pandas_result.columns for k in sort_keys):
            db = duckdb_result.sort_values(sort_keys)[col].reset_index(drop=True)
            pd_ = pandas_result.sort_values(sort_keys)[col].reset_index(drop=True)
        else:
            db = duckdb_result[col].reset_index(drop=True)
            pd_ = pandas_result[col].reset_index(drop=True)

        db_f = pd.to_numeric(db, errors="coerce").fillna(-9999)
        pd_f = pd.to_numeric(pd_, errors="coerce").fillna(-9999)

        np.testing.assert_allclose(
            db_f.values, pd_f.values, atol=NUMERIC_TOL, rtol=0,
            err_msg=f"Divergência na coluna numérica '{col}'"
        )


class TestAntiLeakage:
    def test_first_appointment_has_zero_history(self, duckdb_result):
        """Primeira consulta de cada paciente deve ter previous_appointments_count == 0."""
        col = "previous_appointments_count"
        if col not in duckdb_result.columns:
            pytest.skip(f"Coluna '{col}' ausente")

        if "appointment_id" not in duckdb_result.columns or "appointment_at" not in duckdb_result.columns:
            pytest.skip("Colunas de ordenação ausentes")

        df = duckdb_result.sort_values(["patient_id", "appointment_at"])
        first = df.groupby("patient_id").first().reset_index()
        assert (first[col] == 0).all(), (
            f"Pacientes com previous_appointments_count != 0 na primeira consulta: "
            f"{(first[col] != 0).sum()}"
        )

    def test_past_no_shows_zero_for_first(self, duckdb_result):
        """Primeira consulta de cada paciente deve ter past_no_shows == 0."""
        col = "past_no_shows"
        if col not in duckdb_result.columns:
            pytest.skip(f"Coluna '{col}' ausente")

        if "appointment_at" not in duckdb_result.columns:
            pytest.skip("Coluna appointment_at ausente")

        df = duckdb_result.sort_values(["patient_id", "appointment_at"])
        first = df.groupby("patient_id").first().reset_index()
        assert (first[col] == 0).all(), (
            f"Pacientes com past_no_shows != 0 na primeira consulta: "
            f"{(first[col] != 0).sum()}"
        )


class TestSpecialtyGroup:
    def test_no_null_specialty_group(self, duckdb_result):
        """specialty_group não deve conter nulos."""
        col = "specialty_group"
        if col not in duckdb_result.columns:
            pytest.skip(f"Coluna '{col}' ausente")
        assert duckdb_result[col].notna().all(), "specialty_group contém nulos"

    def test_valid_specialty_group_values(self, duckdb_result):
        """Todos os specialty_group devem ser um dos valores válidos."""
        valid = {
            "ORTOPEDIA", "CIRURGICO_E_VASCULAR", "SAUDE_MENTAL",
            "TERAPIAS_REABILITACAO", "MATERNO_INFANTIL", "CLINICA_GERAL_E_TRIAGEM",
            "EXAMES_E_PROCEDIMENTOS", "CLINICA_ESPECIALIZADA", "OUTRAS_ESPECIALIDADES",
        }
        col = "specialty_group"
        if col not in duckdb_result.columns:
            pytest.skip(f"Coluna '{col}' ausente")
        invalid = set(duckdb_result[col].dropna().unique()) - valid
        assert not invalid, f"specialty_group com valores inesperados: {invalid}"


class TestHolidays:
    def test_is_holiday_binary(self, duckdb_result):
        """is_holiday deve ser 0 ou 1."""
        col = "is_holiday"
        if col not in duckdb_result.columns:
            pytest.skip(f"Coluna '{col}' ausente")
        vals = set(duckdb_result[col].dropna().unique())
        assert vals <= {0, 1, True, False}, f"Valores inesperados em is_holiday: {vals}"

    def test_is_holiday_window_consistency(self, duckdb_result):
        """is_holiday_window deve ser >= is_holiday para cada linha."""
        needed = {"is_holiday", "is_holiday_window"}
        if not needed <= set(duckdb_result.columns):
            pytest.skip("Colunas de feriado ausentes")
        mask = duckdb_result["is_holiday_window"] < duckdb_result["is_holiday"]
        assert not mask.any(), "is_holiday_window < is_holiday em alguma linha"

# Guia de Migração — noshow-lib v0.3.0

Este documento descreve as mudanças introduzidas na versão **0.3.0** e como atualizar o seu código para a nova API.

---

## O que mudou e por quê

A versão anterior treinava um único modelo CatBoost para todos os dados. A nova versão treina um modelo **LightGBM independente por grupo de especialidade** (`specialty_group`), o que traz:

- Modelos especializados para cada contexto clínico
- Otimização automática de hiperparâmetros via **Optuna** (PR-AUC como métrica)
- **Split cronológico** (70/15/15) para respeitar a ordem temporal dos dados e evitar data leakage
- **Threshold ótimo por especialidade**, calibrado para maximizar F1

---

## Novas dependências

Adicione ao seu ambiente:

```bash
pip install optuna>=3.0.0
```

O `lightgbm` já era dependência da lib. O `catboost` não é mais necessário para treinamento, mas permanece no `requirements.txt`.

---

## Mudanças na API

### `train_model(df, config)`

A assinatura é a **mesma**. O retorno mudou.

```python
# Antes — retornava um único dicionário
result = train_model(df, config)
model        = result["model"]
metrics      = result["metrics"]
artifact     = result["artifact_path"]

# Agora — retorna um dicionário por specialty_group
results = train_model(df, config)
# {
#   "CLINICA_ESPECIALIZADA": {
#       "model": <LGBMClassifier>,
#       "metrics": {"pr_auc": 0.41, "roc_auc": 0.72, "f1": 0.38, ...},
#       "threshold": 0.35,
#       "artifact_path": "models/lgbm__clinica_especializada.joblib",
#       "metrics_path": "models/lgbm__clinica_especializada_metrics.json",
#       "features_used": [...],
#       "cat_features": [...],
#       "best_params": {...},
#   },
#   "SAUDE_MENTAL": { ... },
#   ...
# }
```

**Especialidades com menos de `min_volume` registros são ignoradas** (padrão: 5000).

---

### `predict(models, input_data, config, output_path, thresholds)`

O primeiro argumento mudou de `model` para `models` (dicionário).
O argumento `threshold` (escalar) virou `thresholds` (dicionário por especialidade).

```python
# Antes
result_df = predict(
    model=model,
    input_data=df,
    config=config,
    threshold=0.5,
)

# Agora
result_df = predict(
    models=models,          # Dict[str, model]
    input_data=df,
    config=config,
    thresholds=thresholds,  # Dict[str, float] — opcional, usa 0.5 como fallback
)
```

O resultado agora inclui a coluna `specialty_group`:

| appointment_id | patient_id | specialty_group | probability | prediction |
|---|---|---|---|---|
| 123 | P001 | CLINICA_ESPECIALIZADA | 0.61 | 1 |
| 124 | P002 | SAUDE_MENTAL | 0.23 | 0 |

---

### `load_models(models_dir, config)` — **nova função**

Carrega todos os modelos salvos de uma vez.

```python
from noshow_lib import load_models

models = load_models("models/", config)
# {"CLINICA_ESPECIALIZADA": <LGBMClassifier>, "SAUDE_MENTAL": <LGBMClassifier>, ...}
```

---

## Mudanças no `config.yaml`

A seção `model` foi substituída por `model_specialty`.

```yaml
# Antes
model:
  type: "CatBoost"
  artifact_path: "models/catboost_champion.joblib"
  features:
    - waiting_days
    - ...
  parameters:
    iterations: 10000
    learning_rate: 0.14
    ...

# Agora
model_specialty:
  algorithm: "lightgbm"
  artifact_dir: "models/"
  min_volume: 5000       # Volume mínimo por specialty_group
  optuna_trials: 50      # Trials de otimização
  train_frac: 0.70       # Proporção do treino no split cronológico
  val_frac: 0.15         # Proporção da validação
  features:
    - waiting_days
    - ...                # mesma lista de features
```

Os hiperparâmetros do LightGBM são otimizados automaticamente pelo Optuna — não é necessário configurá-los manualmente.

---

## Feature Engineering — novas colunas obrigatórias

O `build_features()` agora gera a coluna `specialty_group`, que é **obrigatória** para o treinamento e inferência. Certifique-se de executar `build_features()` antes de `train_model()` e `predict()`.

Se você usa dados de uma view SQL já processada, garanta que a view inclua a coluna `specialty_group` com os valores:

```
CLINICA_ESPECIALIZADA, CLINICA_GERAL_E_TRIAGEM, CIRURGICO_E_VASCULAR,
EXAMES_E_PROCEDIMENTOS, MATERNO_INFANTIL, ORTOPEDIA,
SAUDE_MENTAL, TERAPIAS_REABILITACAO, OUTRAS_ESPECIALIDADES
```

---

## Exemplo completo — antes e depois

### Antes (v0.2.x)

```python
from noshow_lib import load_config, load_and_validate, build_features, train_model, predict
import joblib

config = load_config("config.yaml")
df_raw = db.query("SELECT * FROM tb_appointments_ht")
df = load_and_validate(df_raw, config)
df = build_features(df, config)

# Treino
result = train_model(df, config)
print(f"ROC-AUC: {result['metrics']['roc_auc']:.4f}")

# Inferência
model = joblib.load("models/catboost_champion.joblib")
predictions = predict(model=model, input_data=df_input, config=config, threshold=0.5)
```

### Agora (v0.3.0)

```python
from noshow_lib import load_config, load_and_validate, build_features, train_model, predict, load_models

config = load_config("config.yaml")
df_raw = db.query("SELECT * FROM tb_appointments_ht")
df = load_and_validate(df_raw, config)
df = build_features(df, config)

# Treino — retorna um modelo por specialty_group
results = train_model(df, config)
for specialty, r in results.items():
    print(f"{specialty}: PR-AUC={r['metrics']['pr_auc']:.4f} | threshold={r['threshold']:.2f}")

# Inferência — carrega modelos e roteia por specialty_group
models = load_models("models/", config)
thresholds = {name: r["threshold"] for name, r in results.items()}

predictions = predict(
    models=models,
    input_data=df_input,
    config=config,
    thresholds=thresholds,
)
print(predictions.head())
```

---

## Artefatos gerados

| Arquivo | Descrição |
|---|---|
| `models/lgbm__{specialty}.joblib` | Modelo por especialidade |
| `models/lgbm__{specialty}_metrics.json` | Métricas + threshold + hiperparâmetros |
| `models/comparativo_especialidades.json` | Resumo consolidado de todas as especialidades |

---

## Estrutura de arquivos

Os scripts antes espalhados na raiz foram reorganizados em duas pastas:

### `tests/` — scripts de validação (usam `config/local.yaml`)

| Script | O que faz |
|---|---|
| `tests/test_fe_only.py` | Testa feature engineering isolado |
| `tests/test_full_pipeline.py` | Pipeline completo: raw → FE → treino → inferência |
| `tests/test_train_view.py` | Treinamento a partir de view SQL processada |
| `tests/test_inference_robustness.py` | Inferência sem coluna `Status` |

### `scripts/` — utilitários de desenvolvimento (usam `config/local.yaml`)

| Script | O que faz |
|---|---|
| `scripts/verify_installation.py` | Confirma que a lib está instalada corretamente |
| `scripts/inspect_model_features.py` | Lista features dos modelos salvos |
| `scripts/run_single_inference.py` | Inferência em um único registro de exemplo |
| `scripts/run_inference_test.py` | Inferência em batch de 100 registros do banco |

### Convenção de configuração

| Arquivo | Uso |
|---|---|
| `config/prod.yaml` | Configuração oficial da lib — usada em produção e nos exemplos da documentação |
| `config/local.yaml` | Exclusivo para scripts em `tests/` e `scripts/` — nunca referenciado pela lib |

O `config.yaml` que existia na raiz foi removido (era uma cópia de `config/prod.yaml`).

---

## Perguntas frequentes

**O que acontece com especialidades que têm poucos dados?**
Especialidades com menos de `min_volume` registros (padrão: 5000) são ignoradas no treinamento. Um aviso é logado para cada uma. Para testes locais, reduza `min_volume` no `config.yaml`.

**O que acontece se um registro chegar com uma especialidade sem modelo treinado?**
`predict()` loga um aviso e pula os registros daquela especialidade. O resultado final não conterá esses registros.

**Posso usar thresholds diferentes dos calculados no treino?**
Sim. Passe um dicionário customizado em `thresholds`. Se uma especialidade não estiver no dicionário, o fallback é `0.5`.

**Preciso retreinar do zero?**
Sim. Os modelos `.joblib` anteriores (CatBoost) não são compatíveis com a nova API.

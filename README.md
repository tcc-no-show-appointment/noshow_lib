# noshow-lib

Biblioteca Python para predição de **no-show em consultas médicas**. Cobre o pipeline completo de ML: validação de dados, feature engineering, treinamento e inferência — com um modelo LightGBM independente por grupo de especialidade.

Desenvolvida como parte do TCC — Engenharia da Computação, Fundação Salvador Arena.

---

## Instalação

**1. Clone o repositório e instale em modo editável:**

```bash
git clone https://github.com/tcc-no-show-appointment/noshow-lib.git
cd noshow_lib
pip install -e .
```

**2. Crie o arquivo `.env` na raiz com as credenciais do banco:**

```env
DB_SERVER=seu_servidor
DB_NAME=nome_do_banco
DB_USER=usuario
DB_PASSWORD=senha
```

---

## Uso rápido

```python
from noshow_lib import load_config, load_and_validate, build_features, train_model, predict, load_models
from config.db import db

# 1. Carregar configuração de produção
config = load_config("config/prod.yaml")

# 2. Extrair dados brutos
df_raw = db.query("SELECT * FROM tb_appointments_ht")

# 3. Validar e gerar features
df = load_and_validate(df_raw, config)
df = build_features(df, config)

# 4. Treinar — um modelo LightGBM por specialty_group
results = train_model(df, config)
for specialty, r in results.items():
    print(f"{specialty}: PR-AUC={r['metrics']['pr_auc']:.4f} | threshold={r['threshold']:.2f}")

# 5. Inferência
models = load_models("models/", config)
thresholds = {name: r["threshold"] for name, r in results.items()}

predictions = predict(models=models, input_data=df_raw, config=config, thresholds=thresholds)
print(predictions.head())
```

---

## Estrutura do projeto

```
noshow_lib/
├── src/noshow_lib/               # Código da biblioteca
│   ├── config.py                 # Carregamento de YAML
│   ├── data_handler.py           # Validação de schema
│   ├── feature_engineering.py   # Pipeline de features (~67 features)
│   ├── logger.py                 # Setup de logs
│   ├── model_training.py         # Treinamento LightGBM + Optuna por specialty_group
│   └── model_inference.py        # Inferência roteada por specialty_group
│
├── config/
│   ├── prod.yaml                 # Configuração oficial da lib (produção)
│   ├── local.yaml                # Configuração para desenvolvimento/testes locais
│   ├── db.py                     # Conexão com SQL Server
│   ├── settings.py               # Leitura do APP_ENV
│   └── readme.md
│
├── tests/                        # Scripts de teste (usam config/local.yaml)
│   ├── test_fe_only.py           # Testa feature engineering isolado
│   ├── test_full_pipeline.py     # Pipeline completo: raw → FE → treino → predict
│   ├── test_train_view.py        # Treinamento via view SQL
│   └── test_inference_robustness.py  # Inferência sem coluna Status
│
├── scripts/                      # Utilitários de desenvolvimento
│   ├── verify_installation.py    # Verifica instalação da lib
│   ├── inspect_model_features.py # Lista features dos modelos salvos
│   ├── run_single_inference.py   # Inferência em registro único
│   └── run_inference_test.py     # Inferência em batch (100 registros)
│
├── models/                       # Artefatos gerados pelo treinamento (não versionado)
│   ├── lgbm__{specialty}.joblib
│   ├── lgbm__{specialty}_metrics.json
│   └── comparativo_especialidades.json
│
├── README.md
├── MIGRATION.md                  # Guia de migração entre versões
├── pyproject.toml
└── requirements.txt
```

---

## Configuração

| Arquivo | Uso |
|---|---|
| `config/prod.yaml` | Configuração oficial. Usada em produção e nos exemplos da documentação. |
| `config/local.yaml` | Configuração local para desenvolvimento. Usada pelos scripts em `tests/` e `scripts/`. |

O ambiente é definido pela variável `APP_ENV` no `.env` (`local` ou `prod`).

---

## API pública

```python
from noshow_lib import (
    load_config,          # Carrega config YAML
    load_and_validate,    # Valida schema do DataFrame
    build_features,       # Gera ~67 features
    train_model,          # Treina um LightGBM por specialty_group
    predict,              # Inferência roteada por specialty_group
    load_models,          # Carrega modelos salvos de um diretório
    setup_logger,         # Logger padronizado
)
```

Consulte o [MIGRATION.md](MIGRATION.md) para detalhes sobre mudanças entre versões.

---

## Scripts disponíveis

### `tests/`
| Script | Como executar | O que faz |
|---|---|---|
| `test_fe_only.py` | `python tests/test_fe_only.py` | Testa feature engineering com 1000 registros do banco |
| `test_full_pipeline.py` | `python tests/test_full_pipeline.py` | Pipeline completo: raw → FE → treino → inferência |
| `test_train_view.py` | `python tests/test_train_view.py` | Treinamento direto a partir de view SQL processada |
| `test_inference_robustness.py` | `python tests/test_inference_robustness.py` | Inferência em registro sem coluna Status |

### `scripts/`
| Script | Como executar | O que faz |
|---|---|---|
| `verify_installation.py` | `python scripts/verify_installation.py` | Confirma que a lib está instalada corretamente |
| `inspect_model_features.py` | `python scripts/inspect_model_features.py` | Lista features de todos os modelos salvos |
| `run_single_inference.py` | `python scripts/run_single_inference.py` | Roda inferência em um registro hardcoded de exemplo |
| `run_inference_test.py` | `python scripts/run_inference_test.py` | Roda inferência em batch de 100 registros do banco |

---

## Autores

- Guilherme Golçaves
- Lohan Batista
- Paulo Henrique
- Rodrigo Puertas

MIT License

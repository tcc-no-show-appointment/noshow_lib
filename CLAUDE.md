# TCC — noshow_lib

## Contexto
Biblioteca Python central do projeto. Compartilhada entre `no-show-training-api` e `no-show-prediction-api`. Cobre o pipeline completo: validação, feature engineering, treinamento e inferência.

## Instalação (dev)
```bash
pip install -e .
```
Requer `.env` na raiz com credenciais do banco (DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD).

## Estrutura
```
src/          → código da biblioteca
models/       → modelos base/schemas
config/       → configurações
tests/        → testes
```

## Responsabilidades principais
- Validação de dados de entrada
- Feature engineering (usado por ambas as APIs via `build_features()`)
- Pipeline de treinamento (modo `external_access`)
- Inferência / predição

## Dependências downstream
Esta lib é usada por:
- `no-show-training-api` — chama pipeline de treino
- `no-show-prediction-api` — chama `build_features()` + inferência

## O que NÃO fazer
- Não quebrar a interface pública sem atualizar as duas APIs que dependem dela
- Mudanças em `build_features()` impactam ambas as APIs — testar nas duas antes de commitar

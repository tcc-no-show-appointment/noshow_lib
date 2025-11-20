# Dados do Projeto

Este diretório organiza os dados utilizados no projeto `noshow-prediction-ml` em diferentes estágios do pipeline de machine learning. A estrutura visa garantir a rastreabilidade e a organização dos dados desde a sua origem até as previsões.

## Subdiretórios:

- `01 - raw/`: Contém os dados brutos originais, exatamente como foram coletados, sem nenhuma modificação.
  - `noshowappointments.csv`: O conjunto de dados original de agendamentos.
- `02 - preprocessed/`: Armazena os dados após as etapas iniciais de limpeza e pré-processamento.
  - `noshowappointments_processed.csv`: Dados após limpeza e pré-processamento.
- `03 - features/`: Contém os dados com as features engenheiradas e selecionadas, prontos para o treinamento do modelo.
  - `features.csv`: Arquivo com as features finais.
  - `noshowappointments_features.csv`: Dados com as features engenheiradas.
- `04 - predictions/`: Guarda os resultados das previsões do modelo e métricas de avaliação.
  - `metrics.csv`: Métricas de avaliação do modelo.

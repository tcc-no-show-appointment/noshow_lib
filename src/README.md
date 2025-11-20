# Código Fonte

Este diretório contém os scripts Python principais para o projeto `noshow-prediction-ml`. Cada script é responsável por uma etapa específica do pipeline de machine learning.

## Arquivos:

- `data_processing.py`: Responsável pelo carregamento inicial, limpeza e pré-processamento dos dados brutos.
- `feature_engineering.py`: Contém funções para a criação de novas features a partir dos dados processados, que são então utilizadas para o treinamento do modelo.
- `model_prediction.py`: Fornece funcionalidade para fazer previsões usando um modelo treinado.
- `model_training.py`: Implementa a lógica para o treinamento de modelos de machine learning, incluindo seleção e avaliação de modelos.
- `run_pipeline.py`: Orquestra todo o pipeline de machine learning, desde o processamento de dados até a previsão do modelo.

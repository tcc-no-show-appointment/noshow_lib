from importlib.metadata import version as get_version

try:
    v = get_version("noshow-lib")
    print(f"Versão da noshow_lib detectada: {v}")
except Exception as e:
    print(f"Não foi possível detectar a versão via metadata: {e}")

try:
    from noshow_lib import setup_logger, load_config, train_model, predict, load_models
    logger = setup_logger("install_test")
    logger.info("Biblioteca importada com sucesso!")

    import noshow_lib
    print(f"Localização da lib: {noshow_lib.__file__}")

except ImportError as e:
    print(f"Erro ao importar a biblioteca: {e}")
except Exception as e:
    print(f"Erro inesperado: {e}")

# src/noshow_lib/config.py

from pathlib import Path
from typing import Union, Dict
import yaml
from yaml import YAMLError

def load_config(path: Union[str, Path]) -> Dict:
    """
    Loads a YAML configuration file and returns it as a dictionary.

    This function handles file opening, parsing, and error management specifically
    for YAML files. If the file is empty, it returns an empty dictionary.

    Args:
        path (str or Path): The file path to the YAML configuration file. 
                            Can be a string or a pathlib.Path object.

    Returns:
        dict: A dictionary containing the configuration parameters loaded from the file.
              Returns an empty dict {} if the file is valid but empty.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        YAMLError: If the file contains invalid YAML syntax.
    """
    path = Path(path)
    
    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        if config is None:
            return {}
            
        return config

    except FileNotFoundError:
        print(f"Error: Configuration file not found at: {path}")
        raise 

    except YAMLError as e:
        print(f"Error: Syntax error in YAML file: {path}")
        print(f"Error details: {e}")
        raise

    except Exception as e:
        print(f"Unexpected error while loading config: {e}")
        raise
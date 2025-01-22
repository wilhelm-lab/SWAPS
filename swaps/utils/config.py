import os
import warnings

from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode as ConfigurationNode
from pathlib import Path

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.


def get_cfg_defaults(singleton: ConfigurationNode):
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return singleton.clone()


def combine_cfgs(
    path_cfg_data: Path,
    path_cfg_override: Path,
    singleton: ConfigurationNode,
):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data = Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override = Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults(singleton)

    # Merge from the path_data
    # Priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    return cfg_base


def update_cfg_using_dotenv() -> list:
    """
    In case when there are dotenvs, try to return list of them.

    # It is returning a list of hard overwrite.
    :return: empty list or overwriting information
    """
    # If .env not found, bail
    if find_dotenv() == "":
        warnings.warn(".env files not found. YACS config file merging aborted.")
        return []

    # Load env.
    load_dotenv(find_dotenv(), verbose=True)

    # Load variables
    list_key_env = {
        "DATASET.RAW_DATA_PATH",
        "DATASET.ACTIVATION_PATH",
        "DATASET.MODEL_DATA_PATH",
    }

    # Instantiate return list.
    path_overwrite_keys = []

    # Go through the list of key to be overwritten.
    for key in list_key_env:
        # Get value from the env.
        value = os.getenv("path_overwrite_keys")

        # If it is none, skip. As some keys are only needed during training and others during the prediction stage.
        if value is None:
            continue

        # Otherwise, adding the key and the value to the dictionary.
        path_overwrite_keys.append(key)
        path_overwrite_keys.append(value)

    return path_overwrite_keys

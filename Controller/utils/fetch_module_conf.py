import sys

sys.path.append("utils")

import configparser
from pathlib import Path


def load_config():
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.absolute() / "utils" / "config.ini"
    config.read(config_path)
    conf_dict = {sect: dict(config.items(sect)) for sect in config.sections()}
    conf_dict.pop("root", None)
    return conf_dict


class ModuleConfiguration:
    def __init__(self):
        conf_dict = load_config()
        self.server_config = conf_dict.get("Server")
        self.experiment_info = conf_dict.get("ExperimentInfo")
        self.external_lb_config = conf_dict.get("External-LoadBalancer")

        self.nginx_basic_status_url: str = "basic_status"

import sys

sys.path.append('utils')
from collections import defaultdict
from dataclasses import field, dataclass
import datetime

import configparser
from pathlib import Path


def load_config():
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.absolute() / 'utils' / 'config.ini'
    config.read(config_path)
    conf_dict = {sect: dict(config.items(sect)) for sect in config.sections()}
    conf_dict.pop('root', None)
    return conf_dict


class ModuleConfiguration:
    def __init__(self):
        conf_dict = load_config()
        self.server_config = conf_dict.get('Server')
        self.experiment_info = conf_dict.get('ExperimentInfo')
        self.external_lb_config = conf_dict.get('External-LoadBalancer')

        self.nginx_basic_status_url: str = "basic_status"


# module_conf = ModuleConfiguration(conf_dict)


class SocialNetworkConfiguration:
    def __init__(self, conf_dict):
        self._mongodb_port_config = conf_dict.get('SocialNetwork-DB-Port')
        self._microservice_port_config = conf_dict.get('SocialNetwork-Microservice-Port')


@dataclass
class FunctionRules:
    cpu_util: float = 40
    cpu_util_default_boolean = True
    cpu_operator: str = "ge"

    memory_util: float = 30
    memory_util_default_boolean = True
    memory_operator: str = "ge"

    arrival_rate: int = 5
    arrival_rate_default_boolean = True
    arrival_rate_operator: str = "ge"


@dataclass
class FunctionWithServiceCandidate:
    service_candidate: str
    rules_for_scaling_policy: dict = field(default_factory=defaultdict)


@dataclass
class ScalingPolicyMetric:
    arrival_rate: float = None
    cpu_util: float = None
    memory_util: float = None


@dataclass
class PreviouslyLaunchedInstancesClass:
    instance_id: str
    running_time: datetime.datetime
    gap_in_seconds: float

import configparser
import datetime
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import boto3

sys.path.append("utils")

from aws_key import CREDENTIALS


# Fetch Configuration
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
        self._server_config = conf_dict.get("Server")
        self._workload_input_config = conf_dict.get("Workload-Input")
        self._external_lb_config = conf_dict.get("External-LoadBalancer")

        self.nginx_basic_status_url: str = "basic_status"


module_conf = ModuleConfiguration()


class VM:
    """
    A class representing an EC2 virtual machine instance.
    """

    def __init__(self, instance_id: str) -> None:
        """
        Initializes a new instance of the VM class.

        Args:
            instance_id (str): The EC2 instance ID.
        """
        self.worker_id_dict: Dict[str, str] = {}
        self.instance_id: str = instance_id
        self.instance_ip: str = self.get_instance_private_ip()
        self.vm_type: str = "ondemand"
        self.weight: int = 1
        self.loadcat_data: defaultdict = defaultdict(dict)

    def set_weight(self, weight: int) -> None:
        """
        Sets the weight of the VM.

        Args:
            weight (int): The weight to set.
        """
        self.weight = weight

    def get_weight(self) -> int:
        """
        Returns the weight of the VM.

        Returns:
            int: The weight of the VM.
        """
        return self.weight

    def get_instance_id(self) -> str:
        """
        Returns the EC2 instance ID.

        Returns:
            str: The EC2 instance ID.
        """
        return self.instance_id

    def add_loadcat_data(self, data_dict: dict, service_name: str) -> None:
        """
        Adds the loadcat data to the VM's data for a service.

        Args:
            data_dict (dict): The loadcat data dictionary.
            service_name (str): The name of the service.
        """
        self.loadcat_data[service_name] = data_dict

    def return_loadcat_data_per_service(self) -> defaultdict:
        """
        Returns the loadcat data for each service on the VM.

        Returns:
            defaultdict: The loadcat data defaultdict.
        """
        return self.loadcat_data

    def set_worker_id(self, worker_id: str, service_name: str) -> None:
        """
        Sets the worker ID for a service on the VM.

        Args:
            worker_id (str): The worker ID to set.
            service_name (str): The name of the service.
        """
        self.worker_id_dict[service_name] = worker_id

    def get_worker_id(self) -> Dict[str, str]:
        """
        Returns the worker ID dictionary for the VM.

        Returns:
            dict: The worker ID dictionary.
        """
        return self.worker_id_dict

    def set_instance_type(self, vm_type: str) -> None:
        """
        Sets the type of the VM.

        Args:
            vm_type (str): The type of the VM.
        """
        self.vm_type = vm_type

    def get_instance_private_ip(self, ip_type="PrivateIpAddress"):
        """
        Returns the public or private IP address of the instance.
        To get the public IP address, set ip_type to "PublicIp".
        """
        ec2 = boto3.resource("ec2", **CREDENTIALS, region_name="us-east-1")
        instance = ec2.Instance(self.instance_id)
        return getattr(instance.network_interfaces_attribute[0], ip_type)

    def get_instance_cpu_util(self):
        """
        Returns the CPU utilization of the instance over the last 3 minutes.
        """
        cloudwatch = boto3.client("cloudwatch", **CREDENTIALS, region_name="us-east-1")
        print(f"instance_id is : {self.instance_id}")
        response = cloudwatch.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[{"Name": "InstanceId", "Value": self.instance_id}],
            StartTime=datetime.datetime.now() - datetime.timedelta(seconds=180),
            EndTime=datetime.datetime.now(),
            Period=30,
            Statistics=["Average"],
        )
        with open(module_conf.pickle_file, "wb") as output:
            pickle.dump(response["Datapoints"], output, pickle.HIGHEST_PROTOCOL)
        for datapoint in response["Datapoints"]:
            print(datapoint)
        return response


@dataclass
class InstanceInfoWithCPUClass:
    inst_id: str
    cpu_util: float


@dataclass
class InstanceInfoClass:
    vm_class_info: VM = None
    running_time: datetime.datetime = datetime.datetime.now()
    recent_instance: bool = False
    number_of_elapsed_minutes: int = 0
    cpu_util: float = 0


@dataclass
class TerminatedInstanceInfoClass(InstanceInfoClass):
    termination_time: datetime.datetime = datetime.datetime.now()


def return_microservices_that_involve_mongodb():
    """
    Make a list of microservices that integrate with MongoDB, for Lambda update (Use for your own experiment
    """
    global module_conf

    return []

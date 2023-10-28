"""
Run workload and scaling policy with compiler

SYNOPSIS
========
::
    python3 controller.py

DESCRIPTION
===========
1. Open Initial Resources
2. Start simulator (fetching traces and use it for request input) and scaling policy
3. When provisioning, use the compiler to make a hybrid case

ENVIRONMENT
==========
Do configuration before running the workload

    USE_CASE_FOR_EXPERIMENTS

    LoadBalancer Configuration
        AMI_ID
        INSTANCE_WORKERS
        WORKLOAD_CHOICE
        NUMBER_OF_INSTANCES
        CPU IDLE PERCENT
        use_case_for_experiments

    Module Configuration

NOTES
Once the workload is deployed, the one of microservices will  initiate Prometheus, which will be responsible for collecting `runq_latency` metrics for each microservice. This data can be subsequently used to feed into a training model, enabling the selection of microservices that are best suited to meet the Service Level Objectives (SLO). During the provisioning phase, a compiler is utilized to transform the microservice into a Lambda function. Additionally, compiler modifies the orchestrator function, transitioning its calls from a VM-based function to Lambda. And controller.py would reroute request to run hybrid code through Loadcat.

The compiler would make a hybrid code. While provisioning, controller.py would reroute the request to run hybrid code through Loadcat.


"""
import asyncio
import base64
import contextlib
import csv
import datetime
import glob
import logging
import logging.handlers
import math
import os
import pickle
import sys
import threading
import time
import urllib
import urllib.parse
import urllib.request
import urllib.request
import zlib
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from shutil import copyfile
from typing import List
from typing import Optional

import aiohttp
import boto3
import numpy as np
import paramiko
import pytz
import requests
from aiohttp_retry import RetryClient, ExponentialRetry
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Assume the compiler code is in current directory
from compiler import process

sys.path.append("utils")
from fetch_logging_configuration import init_logger, empty_log_file
from fetch_module_conf import ModuleConfiguration
from fetch_vm_related_functionality import (
    InstanceInfoClass,
    InstanceInfoWithCPUClass,
    TerminatedInstanceInfoClass,
    return_microservices_that_involve_mongodb,
)
from make_param_for_composepost import generate_input_for_compose_post_service

from aws_key import CREDENTIALS

# Fetch logger instance and directory to save log file
logger, result_log_with_dir = init_logger()

# Empty log file before running the experiment
empty_log_file()

conf_dict = ModuleConfiguration()

# request & client settings
s = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
s.mount("http://", HTTPAdapter(max_retries=retries))
user_agent = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/51.0.2704.103 Safari/537.36"
}
headers = {**user_agent}

# aws clients
ec2_client = boto3.client("ec2", region_name="us-east-1", **CREDENTIALS)
lambda_client = boto3.client("lambda", region_name="us-east-1", **CREDENTIALS)
cloudwatch_client = boto3.client("cloudwatch", **CREDENTIALS, region_name="us-east-1")

# Semaphore for adding ec2 info
instance_workers_lock = threading.Lock()

# Instance Information
whole_instance_info = defaultdict(InstanceInfoClass)
worker_list_in_lbs = []
instance_list_in_lbs = []
ec2_private_ip = []
instance_to_worker = {}

# Data to save in pickle after creation of initial new instances
data_to_save_in_pickle = {
    "whole_instance_info": whole_instance_info,
    "instance_list_in_lbs": instance_list_in_lbs,
    "worker_list_in_lbs": worker_list_in_lbs,
    "ec2_private_ip": ec2_private_ip,
    "instance_to_worker": instance_to_worker,
}

# List of initial instances or instances from the past 1 minute (for scaling-in)
initial_or_recent_instances = []

# Dictionary to keep track of terminated instance information
terminated_instance_info = defaultdict(InstanceInfoClass)

# Load balancer information
lb_addr = conf_dict.external_lb_config.get("external-loadbalancer-addr")
lb_port = conf_dict.external_lb_config.get("external-loadbalancer-port")

# Info about requests
duration_info = {
    "total_requests": 0,
    "accumulated_request_list_every_minute": [0],
    "duration_list": [],
    "lambda_duration_list": [],
    "duration_in_seconds": 0,
    "violated_duration_list": [],
    "number_of_violation_from_lambda": 0,
    "number_of_violation": 0,
}

# Index for stopping policy (after all the responses are received)
index_for_stopping_scaling_policy = False

# provisioning method (vm or lambda or MicroBlend)
provisioning_method = "vm"

# request type (vm or lambda)
request_type = "vm"

# Fetch lambda that interact with MongoDB (For access to MondoDB in VM)
lambda_arn_list_for_mongodb = return_microservices_that_involve_mongodb()


def configure_experiment_variables():
    """
    Configure experiment variables
    Change the value of variables based on the experiment case
    :return file name to save the result
    """
    global index_for_stopping_scaling_policy, provisioning_method

    experiment_case = conf_dict.experiment_info.get("experiment_case")
    trace_input = conf_dict.experiment_info.get("trace_input")
    request_handle_type = "vm"
    over_provision = False
    result_file_name = None
    index_for_stopping_scaling_policy = False

    experiment_case_list = [
        "all_vm",
        "all_lambda",
        "overprovision",
        "microblend",
        "test_vm",
        "test_lambda",
    ]

    if experiment_case not in experiment_case_list:
        logger.info("Wrong use case input")
        sys.exit(1)

    if experiment_case == "overprovision":
        over_provision = True
        result_file_name = f"{trace_input.split('.')[0]}_over_provision"

    elif experiment_case == "all_lambda":
        request_handle_type = "lambda"

    elif experiment_case == "microblend":
        provisioning_method = "microblend"
        result_file_name = f"{trace_input.split('.')[0]}_microblend"

    # Result file name to save
    result_file_name = (
            result_file_name or f"{trace_input.split('.')[0]}_{request_handle_type}"
    )
    logger.info(f"Result log file name to save : {result_file_name}")
    logger.info(f"Experiment Case : {experiment_case}")

    # Request type
    if experiment_case == "microblend":
        logger.info(
            f"Initial Request Type : '{request_handle_type}' -> using 'lambda' while provisioning"
        )
    else:
        logger.info(f"Request Type : {request_handle_type}\n")

    return result_file_name


# Fetch experiment variables
result_log_file_name = configure_experiment_variables()


def encode_native_object(obj):
    """
    Encode a Python object using pickle, zlib, and base64.

    Args:
        obj: A Python object.

    Returns:
        A string containing the encoded object.
    """
    obj = pickle.dumps(obj)
    obj = zlib.compress(obj)
    obj = base64.b64encode(obj).decode().replace("/", "*")
    return obj


def decode_native_object(encoded_obj):
    """
    Decode a string containing a Python object encoded using `encode_native_object()`.

    Args:
        encoded_obj: A string containing the encoded object.

    Returns:
        The decoded Python object.
    """
    encoded_obj = encoded_obj.replace("*", "/")
    encoded_obj = base64.b64decode(encoded_obj)
    encoded_obj = zlib.decompress(encoded_obj)
    return pickle.loads(encoded_obj)


def disable_workers_in_loadcat(initial_work_id_list: Optional[list] = None):
    """
    Make all workers unavailable in all microservice load_balancers (loadcat)
    """
    logger.info("Make servers unavailable in loadcat")

    # Fetch initial work IDs
    initial_work_id_list = initial_work_id_list or []

    # Get LoadBalancer configuration
    loadbalancer_addr = conf_dict.external_lb_config.get("loadbalancer-addr")
    logger.debug(f"loadbalancer_addr : {loadbalancer_addr}")

    # Get LoadBalancer ID
    lb_id = conf_dict.external_lb_config.get("loadbalancer_id")

    # Fetch worker IDs from servers and exclude initial_work_id_list
    url_to_request = f"http://{loadbalancer_addr}:26590/balancers/{lb_id}"

    worker_id_list_from_lb = []

    # Fetch worker IDs from load balancer
    try:
        html_page = urllib.request.urlopen(url_to_request, timeout=15)
        time.sleep(0.01)
    except Exception as exc:
        logging.error(f"{exc} happened for {url_to_request}")
    else:
        soup = BeautifulSoup(html_page, "html.parser")
        for link in soup.findAll("a"):
            if str(link.get("href")).startswith("/servers/"):
                split_href = str(link.get("href")).split("/")
                worker_id_list_from_lb.extend(
                    split_href[i + 1]
                    for i, c in enumerate(split_href)
                    if c == "servers" and split_href[i + 1] not in initial_work_id_list
                )

    # Make data for every worker ID
    worker_info_dict = {
        each_worker: {
            "settings.address": "0.0.0.0",
            "settings.weight": 1,
            "settings.availability": "unavailable",
            "label": f"{each_worker}_unused",
        }
        for each_worker in worker_id_list_from_lb
        if each_worker not in initial_work_id_list
    }

    # Send POST Method to make them unavailable
    for each_worker_url, worker_data in worker_info_dict.items():
        url_to_append = f"/servers/{each_worker_url}/edit"
        url_for_editing_server = f"http://{loadbalancer_addr}:26590{url_to_append}"
        s.post(url_for_editing_server, worker_data, headers=user_agent)
        time.sleep(0.01)

    logger.info("Finished making workers in loadcat unavailable\n")


def terminate_previous_instances(inst_id_list=None):
    """Terminate previous instances"""
    logger.info("Start terminating previous instances")

    # Find instances with tag name
    server_tag_name = conf_dict.server_config.get("tag_name")

    inst_id_list = inst_id_list or []

    custom_filter = [
        {"Name": "tag:Name", "Values": [server_tag_name]},
        {"Name": "instance-state-name", "Values": ["running"]},
    ]

    try:
        # Find instance id and terminate except initial instances
        response = ec2_client.describe_instances(Filters=custom_filter)
        for res in response["Reservations"]:
            for i in res["Instances"]:
                each_inst_id = i["InstanceId"]
                if each_inst_id in inst_id_list:
                    logger.info(f"\tSkip Initial Instance - {each_inst_id}")
                    continue
                else:
                    terminate_instance(each_inst_id)
        if not response["Reservations"]:
            raise ValueError(f"No instances of {server_tag_name} running")
    except Exception as e:
        logger.info(f"\t{e}")
    else:
        logger.info("Finished terminating instances beforehand\n")


def terminate_instance(instance_id):
    """
    terminate instances with ec2_id
    """
    global ec2_client, logger
    ec2_client.terminate_instances(InstanceIds=[instance_id])
    logger.info(f"\tTerminating instance - {instance_id}")


def create_instance(num, instance_type, image_id, security_g, snid, key_name):
    # Get server tag name
    server_tag_name = conf_dict.server_config.get("tag_name")

    # Create boto3 session
    ec2 = boto3.resource("ec2", region_name="us-east-1", **CREDENTIALS)

    # Create instance from boto3 session
    instance = ec2.create_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        KeyName=key_name,
        MaxCount=num,
        MinCount=1,
        Monitoring={"Enabled": True},
        SecurityGroupIds=[security_g],
        DisableApiTermination=False,
        InstanceInitiatedShutdownBehavior="stop",
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": server_tag_name}],
            }
        ],
    )
    return [instance[i].instance_id for i in range(len(instance))]


def launch_ec2_and_add_to_loadbalancer(num_of_workers_to_add=0, init_phase=False):
    """
    Spawn EC2 instances and add them to load balancers
    """
    global whole_instance_info, instance_list_in_lbs

    logger.info("Start Spawn EC2 and Add them to LoadBalancer")

    # Determine number of instances to launch
    num_of_instances = (
        conf_dict.server_config.get("init_number_of_instances")
        if init_phase
        else num_of_workers_to_add
    )
    logger.info(f"Requesting {num_of_instances} instances")

    # Create ec2 instances
    inst_id_list_to_launch = create_instance(
        num=int(num_of_instances),
        instance_type=conf_dict.server_config.get("instance_type"),
        image_id=conf_dict.server_config.get("ami_id"),
        snid=conf_dict.server_config.get("subnet_id"),
        security_g=conf_dict.server_config.get("security_group"),
        key_name=conf_dict.server_config.get("key_name"),
    )
    logger.info(f"Instances to launch: {inst_id_list_to_launch}")

    # Add instances to load balancer
    logger.info("Start adding EC2 instances to LoadBalancers")

    if index_for_stopping_scaling_policy:
        logger.info("Not adding instance info since workload ended [Ended Policy]\n")
        return

    # Add instances to LB concurrently
    threads = []
    for inst_id in inst_id_list_to_launch:
        instance_list_in_lbs.append(inst_id)
        each_thread = threading.Thread(
            target=add_instance_to_lb, args=[inst_id, init_phase]
        )
        threads.append(each_thread)
        each_thread.start()

    for thread in threads:
        thread.join()

    logger.info(
        f"Finished launching and adding {len(inst_id_list_to_launch)} instances to LoadBalancers\n"
    )


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
        self.instance_id: str = instance_id
        self.instance_ip: str = (
            self.get_instance_private_ip() if instance_id != "test" else "test"
        )
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

    def add_loadcat_data(self, data_dict: dict) -> None:
        """
        Adds the loadcat data to the VM's data for a service.

        Args:
            data_dict (dict): The loadcat data dictionary.
        """
        self.loadcat_data["loadcat_server"] = data_dict

    def return_loadcat_data_per_service(self) -> defaultdict:
        """
        Returns the loadcat data for each service on the VM.

        Returns:
            defaultdict: The loadcat data defaultdict.
        """
        return self.loadcat_data

    def get_instance_private_ip(self, ip_type="PrivateIpAddress"):
        """
        Returns the public or private IP address of the instance.
        To get the public IP address, set ip_type to "PublicIp".
        """
        # keep checking until we get instance id

        ec2 = boto3.resource("ec2", **CREDENTIALS, region_name="us-east-1")
        instance = ec2.Instance(self.instance_id)
        ip_address = ""
        while not ip_address:
            with contextlib.suppress(ClientError):
                ip_address = instance.network_interfaces_attribute[0].get(
                    "PrivateIpAddress"
                )
        return ip_address


def find_public_ip_and_launch_time(ec2_instance_id):
    """Find the public IP address and launch time of an EC2 instance"""

    def utc_to_local(utc_dt):
        local_tz = pytz.timezone("Asia/Seoul")
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
        return local_tz.normalize(local_dt)  # .normalize might be unnecessary

    global ec2_client
    instance_public_ip, ec2_instance_launch_time = None, None
    while True:
        try:
            response1 = ec2_client.describe_instances(InstanceIds=[ec2_instance_id])
            for reservation in response1["Reservations"]:
                for instance in reservation["Instances"]:
                    instance_public_ip = instance.get("PublicIpAddress")
                    ec2_instance_launch_time = instance.get("LaunchTime")
                    if instance_public_ip is None:
                        time.sleep(1)
                        raise TypeError
        except Exception:
            logger.info(f"\t\tNo information of {ec2_instance_id} yet, retrying")
            continue
        break

    return instance_public_ip, utc_to_local(ec2_instance_launch_time).replace(
        tzinfo=None
    )


def check_ssh_for_spawning_time(ip_address):
    """
    Check if SSH is available for the given IP address
    """
    ssh = paramiko.client.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
    key_dir = Path(__file__).parent.absolute() / "utils"
    key_name = conf_dict.server_config.get("key_name")
    key_name_with_dir = f"{key_dir}/{key_name}"
    logger.info(f"\t\tssk key is : {key_name_with_dir}")

    try:
        ssh.connect(ip_address, username="ec2-user", key_filename=key_name_with_dir)
        return datetime.datetime.now()
    except Exception as e:
        logger.info(f"\t\tSSH exception for ip_address: {ip_address} - {e}")
        return False


def add_instance_to_lb(inst_id, init_phase=False):
    """
    Add instances to Loadbalancer per microservice loadbalancer
    """

    global whole_instance_info, logger, ec2_private_ip

    logger.info(f"\t\tinstance id is {inst_id}")

    # Make VM class from instance and set weight as 1 (for loadbalancer)
    server_information = VM(instance_id=inst_id)
    server_information.set_weight(weight=1)

    experiment_case = conf_dict.experiment_info.get("experiment_case")

    # If using Lambda, assume VMs are ready before attaching to LB
    # This lets us assume we already have newly launched instances ready when calculating the number of servers
    if experiment_case == "microblend" and not init_phase:
        instance_info = InstanceInfoClass(running_time=datetime.datetime.now())
        logger.debug(
            "\t\t\t[microblend] -> Add instance since lambda take cares of request upon autoscaling"
        )
        update_instance_info_and_save_to_whole_info(
            instance_info, server_information, inst_id
        )

    # Get public IP address and launch time of instance
    instance_public_address, instance_launch_time = find_public_ip_and_launch_time(
        inst_id
    )
    logger.info(f"\t\tpublic address for {inst_id} is {instance_public_address}")

    # Check if instance is in running state (check if it can be accessed by SSH)
    logger.info(f"\t\tCheck if {inst_id} can be accessed by SSH.")
    while True:
        ssh_success_time = check_ssh_for_spawning_time(instance_public_address)
        time.sleep(0.5)  # Needs some interval, otherwise error happens
        if ssh_success_time:
            logger.info(f"\t\t\t{inst_id} is in running state")
            # Make InfoClass with datetime for instance and save in whole_instance_info
            instance_info = InstanceInfoClass(running_time=datetime.datetime.now())
            break

    logger.info(f"\tStart adding instance {inst_id} to LB")

    # Create and start a new LBAttacher thread for this server information
    attacher_thread = LBAttacher(vm=server_information)
    attacher_thread.start()

    # Wait for the thread to finish executing (i.e., wait for the server to be added to the load balancer)
    attacher_thread.join()

    # Add information to whole_info after being added to LBAttacher
    logger.info(f"\t\tAdding {inst_id} instance info class to whole_instance_info")
    update_instance_info_and_save_to_whole_info(
        instance_info, server_information, inst_id
    )

    # Add private ip for lambda environment (for interacting with MongoDB)
    ec2_private_ip.append(server_information.instance_ip)


def update_instance_info_and_save_to_whole_info(
        instance_info: InstanceInfoClass, server_information: VM, inst_id, hybrid=False
):
    """
    Store information -> Save to whole_instance_info
    """
    global logger

    instance_info.vm_class_info = server_information

    # If we use microblend approach, we assume newly launched instance is already there
    if hybrid:
        instance_info.recent_instance = False
    else:
        instance_info.recent_instance = True

    # Initial CPU utilization is 0
    instance_info.cpu_util = 0
    instance_info.running_time = datetime.datetime.now()

    if index_for_stopping_scaling_policy:
        logger.info("\tNot adding instance info since workload ended [Ended Policy]\n")
        return
    else:
        # Add information to whole_info
        whole_instance_info[inst_id] = instance_info


def attach_server_to_lb(server_info):
    """Attach server to load balancer in a separate thread."""
    LBAttacher(vm=server_info).start()


class LBAttacher(threading.Thread):
    """
    Adding workers to Microservice LoadBalancers
    """

    def __init__(self, vm):
        super(LBAttacher, self).__init__()
        self.vm = vm

    def run(self):
        """
        Add workers to loadbalancers
        """
        add_each_worker_to_lb(self.vm)


def update_worker_attribute(
        balancer_ip_addr_with_port,
        vm,
        worker_id,
        attr_dict=None,
        new_resource=False,
        remove_worker=False,
):
    """
    Update worker's attribute (e.g. availability) in Loadcat
    """

    def get_url_for_edit_server_info(worker_id_from_loadcat):
        """
        Get URL for server edit
        """
        return (
            f"http://{balancer_ip_addr_with_port}/servers/{worker_id_from_loadcat}/edit"
        )

    # If no data for server, make empty one
    if attr_dict is None:
        attr_dict = {}

    if new_resource:
        # Wait some time fpr spinning up new VMs
        logger.info("Wait for VM to be ready (20s)")
        time.sleep(20)

        # Address to edit server's information
        address_for_editing = get_url_for_edit_server_info(worker_id)

        # Fetch server information
        info_for_server = attr_dict.copy()
        info_for_server["settings.availability"] = ["available"]

        # POST to update the server information
        response = s.post(
            url=address_for_editing, data=info_for_server, headers=user_agent
        )
        time.sleep(0.01)
        assert response.status_code == 200

        # Add loadcat data to VM
        vm.add_loadcat_data(info_for_server)

    elif remove_worker:
        # Get address for editing
        address_for_editing = get_url_for_edit_server_info(worker_id)

        # Update status as backup to finish already given request
        info_for_server_dict: dict = vm.return_loadcat_data_per_service().get(
            "loadcat_server"
        )
        info_for_server_dict["settings.availability"] = ["backup"]
        response = s.post(url=address_for_editing, data=info_for_server_dict)
        time.sleep(0.01)
        assert response.status_code == 200

        # Wait some time to finish request that was already sent
        time.sleep(5)
        logger.info("\t\tGive grace period to finish already sent request (5 mins)")

        # Update to unavailable
        info_for_server_dict["settings.availability"] = ["unavailable"]
        response = s.post(
            url=address_for_editing, data=info_for_server_dict, headers=user_agent
        )
        time.sleep(0.01)
        assert response.status_code == 200

    logger.info(f"\t\t{vm.instance_id} updated to {vm.loadcat_data}")


class LBDetacher(threading.Thread):
    """
    Remove workers from microservice loadbalancers and terminate instances
    """

    def __init__(self, vm):
        super(LBDetacher, self).__init__()
        self.vm = vm

    def run(self):
        """
        Remove workers from loadbalancers and terminate instances
        """

        # Remove workers from loadbalancers
        logger.info(
            f"\tStart removing workers of {self.vm.get_instance_id()} from microservice loadbalancers"
        )
        loadbalancer_ip_addr_with_port = conf_dict.external_lb_config[
            "loadbalancer-addr"
        ]
        loadcat_addr_with_port = f"{loadbalancer_ip_addr_with_port}:26590"
        worker_id = instance_to_worker[self.vm.get_instance_id()]
        update_worker_attribute(
            loadcat_addr_with_port, self.vm, worker_id, remove_worker=True
        )
        logger.info(
            f"\tFinished removing workers of {self.vm.get_instance_id()} from microservice loadbalancers"
        )

        # Terminate instance
        logger.info(f"\tStart terminating instance {self.vm.get_instance_id()}")
        terminate_instance(self.vm.get_instance_id())
        logger.info(f"\tFinished terminating instance {self.vm.get_instance_id()}")


def add_each_worker_to_lb(vm: VM):
    """
    Add a new worker to the load balancer

    Steps:
    1. Generate the URL for adding a new server
    2. Send a request to create the new server
    3. Get the server ID from the response URL
    4. Add the server ID to the global worker list
    5. Store the instance ID to server ID mapping in a global dictionary
    6. Add data to the new server (weight, availability, etc.)
    7. After a period of time, update the new server's availability to "available"
    """
    global worker_list_in_lbs, instance_to_worker

    loadbalancer_id = conf_dict.external_lb_config["loadbalancer_id"]
    loadbalancer_ip_addr_with_port = conf_dict.external_lb_config["loadbalancer-addr"]
    loadcat_addr_with_port = f"{loadbalancer_ip_addr_with_port}:26590"
    addr_for_new_worker = f"/balancers/{loadbalancer_id}/servers/new"

    addr_for_url = f"http://{loadcat_addr_with_port}{addr_for_new_worker}"
    logger.info(f"\t\tAdding {vm.get_instance_id()} to load balancer {loadbalancer_id}")

    # Data to put in Loadcat
    port_number = int(conf_dict.server_config.get("server_port"))
    info_for_server = {
        "settings.address": f"{vm.get_instance_private_ip()}:{port_number}",
        "label": vm.get_instance_id(),
    }

    # Make request to create new server
    try:
        r = s.post(url=addr_for_url, data=info_for_server, headers=user_agent)
        time.sleep(0.01)
    except requests.exceptions.ConnectionError as e:
        logger.info(f"Exception {e} happened")
        return

    # get server id from url response
    assert r.status_code == 200
    server_id = r.url.split("/servers/")[1].split("/")[0]

    # Add worker id to global worker list
    worker_list_in_lbs.append(server_id)
    instance_to_worker[vm.get_instance_id()] = server_id

    # Put data in dictionary
    info_for_server["settings.weight"] = 1
    info_for_server["settings.availability"] = "unavailable"

    # Add other information to Loadcat
    url_to_append = f"/servers/{server_id}/edit"
    url = f"http://{loadcat_addr_with_port}{url_to_append}"
    response = s.post(url=url, data=info_for_server, headers=user_agent)
    assert response.status_code == 200

    # After a period of time, make server available (due to VM's launching time)
    update_worker_attribute(
        balancer_ip_addr_with_port=loadcat_addr_with_port,
        vm=vm,
        worker_id=server_id,
        attr_dict=info_for_server,
        new_resource=True,
    )


def remove_each_worker_from_lb(vm: VM):
    """Remove worker from load balancer (making it unavailable)"""
    loadbalancer_ip_addr_with_port = conf_dict.external_lb_config["loadbalancer-addr"]
    loadcat_addr_with_port = f"{loadbalancer_ip_addr_with_port}:26590"
    worker_id = instance_to_worker[vm.get_instance_id()]

    update_worker_attribute(
        balancer_ip_addr_with_port=loadcat_addr_with_port,
        vm=vm,
        worker_id=worker_id,
        remove_worker=True,
    )


def update_ec2_private_ip_env_in_lambda(lambda_arn_list):
    """
    This function is for code where Lambda interacts with MongoDB
    Iterate lambda microservices and update mongo db for load balancing"""

    def update_mongo_addr_list_in_lambda(lambda_arn, server_addr_str):
        """Invoke Lambda"""
        lambda_client.update_function_configuration(
            FunctionName=lambda_arn,
            Environment={"Variables": {"mongo_addr_list": server_addr_str}},
        )

    logger.info("Start Updating Private Ip ENV for Mongo in Lambda")

    # Make a format <ip>;<ip>
    private_ip_addr_str = ";".join([str(elem) for elem in ec2_private_ip])
    logger.info(f"\tPrivate ip address is {private_ip_addr_str}")

    # Make workers from each microservice unavailable concurrently
    threads = []
    for each_lambda_arn in set(lambda_arn_list):
        thread = threading.Thread(
            target=update_mongo_addr_list_in_lambda,
            args=[each_lambda_arn, private_ip_addr_str],
        )
        threads.append(thread)
        threads[-1].start()
    for thread in threads:
        thread.join()

    logger.info("Finished Updating Private Ip ENV for Mongo in Lambda")


async def make_compose_post_input_dict():
    """
    Generates random input for the Compose Post service and returns a dictionary containing the parameters.

    Returns:
        dict: A dictionary containing the parameters for the Compose Post service.
    """

    # Generate random input and encode params
    compose_post_parameter = generate_input_for_compose_post_service()
    req_id = compose_post_parameter.req_id
    username, user_id = compose_post_parameter.username, compose_post_parameter.user_id
    text, media_ids = compose_post_parameter.text, compose_post_parameter.media_ids
    media_types, post_type = (
        compose_post_parameter.media_types,
        compose_post_parameter.post_type,
    )
    return {
        "req_id": req_id,
        "username": username,
        "user_id": user_id,
        "text": text,
        "media_ids": media_ids,
        "media_types": media_types,
        "post_type": post_type,
        "carrier": {},
    }


async def send_request(sess, num=0, arrive_time=0, num_of_tasks=0, total_reqs=0):
    duration_info["duration_in_seconds"] = arrive_time

    slo_target = float(conf_dict.experiment_info.get("slo_target"))

    compose_post_input_dict = await make_compose_post_input_dict()
    encoded_input_dict = encode_native_object(compose_post_input_dict)

    url = f"http://{lb_addr}:{lb_port}/python-social-network"
    url = "http://httpbin.org/get"  # for testing purposes
    await asyncio.sleep(0.001)

    start = time.time()
    if request_type == "vm":
        try:
            async with sess.get(url, timeout=5) as resp:
                assert resp.status == 200

                duration = float((time.time() - start) * 1000)

                if duration <= slo_target:
                    await save_and_print_response(
                        arrive_time, duration, num, num_of_tasks, total_reqs
                    )
                else:
                    await save_and_print_response(
                        arrive_time,
                        duration,
                        num,
                        num_of_tasks,
                        total_reqs,
                        violated=True,
                    )

        except asyncio.TimeoutError:
            logger.info("Exception -> Timeout Error")
            duration = float(5000)
            await save_and_print_response(
                arrive_time, duration, num, num_of_tasks, total_reqs, violated=True
            )

        except aiohttp.ClientConnectionError as e:
            logger.info(f"Exception {e} -> ClientConnectionError -> Violated SLO")
            duration = float(5000)
            await save_and_print_response(
                arrive_time, duration, num, num_of_tasks, total_reqs, violated=True
            )

        except Exception as e:
            logger.info(f"Other exception -> Violated SLO: {e}")
            duration = float(5000)
            await save_and_print_response(
                arrive_time, duration, num, num_of_tasks, total_reqs, violated=True
            )

    elif request_type == "lambda":

        try:

            lambda_url = "https://lambda.us-east-1.amazonaws.com/xxxxx"
            # call Lambda using the with
            async with sess.get(lambda_url, timeout=5) as resp:
                assert resp.status == 200

            # Get duration & Add all duration to total duration
            duration = float((time.time() - start) * 1000)

            # When response is less than SLO
            if duration < slo_target:
                await save_and_print_response(
                    arrive_time,
                    duration,
                    num,
                    num_of_tasks,
                    total_reqs,
                    service="lambda",
                )

            else:
                await save_and_print_response(
                    arrive_time,
                    duration,
                    num,
                    num_of_tasks,
                    total_reqs,
                    service="lambda",
                    violated=True,
                )

        except asyncio.TimeoutError:
            logger.info("Exception -> Timeout Error")
            duration = float(5000)
            await save_and_print_response(
                arrive_time,
                duration,
                num,
                num_of_tasks,
                total_reqs,
                service="lambda",
                violated=True,
            )

        except aiohttp.ClientConnectionError as e:
            logger.info(f"Exception {e} -> ClientConnectionError -> Violated SLO")
            duration = float(5000)
            await save_and_print_response(
                arrive_time,
                duration,
                num,
                num_of_tasks,
                total_reqs,
                service="lambda",
                violated=True,
            )

        except Exception:
            logger.info("Other exception -> Violated SLO")
            duration = float(5000)
            await save_and_print_response(
                arrive_time,
                duration,
                num,
                num_of_tasks,
                total_reqs,
                service="lambda",
                violated=True,
            )


async def save_and_print_response(
        arrival_time,
        duration,
        request_num,
        num_of_tasks,
        total_requests,
        service="vm",
        violated=False,
):
    """Save and print response information"""
    duration_info["duration_list"].append(duration)

    if service == "vm":
        if violated:
            duration_info["number_of_violation"] += 1
            duration_info["violated_duration_list"].append(duration)
        logger.info(
            f"VM - Total requests: {total_requests} - "
            f"Request {request_num + 1}/{num_of_tasks} at time {arrival_time} - "
            f"Response: null, Duration: {duration} - "
            f"Violated SLOs: {duration_info['number_of_violation']} - "
            f"Total duration list size: {len(duration_info['duration_list'])}"
        )
    if service == "lambda":
        duration_info["lambda_duration_list"].append(duration)

        if violated:
            duration_info["number_of_violation"] += 1
            duration_info["violated_duration_list"].append(duration)
            duration_info["number_of_violation_from_lambda"] += 1

            logger.info(
                f"Lambda - Total requests: {total_requests} - "
                f"Request {request_num + 1}/{num_of_tasks} at time {arrival_time} - "
                f"Response duration: {duration} - "
                f"Violated SLOs: {duration_info['number_of_violation']} - "
                f"Total duration list size: {len(duration_info['duration_list'])}"
            )
        else:
            logger.info(
                f"Lambda - Total requests: {total_requests} - "
                f"Request {request_num + 1}/{num_of_tasks} at time {arrival_time} - "
                f"Response duration: {duration} - "
                f"Violated SLOs: {duration_info['number_of_violation']} - "
                f"Total duration list size: {len(duration_info['duration_list'])})"
            )


def start_workload():
    """Start workload"""
    asyncio.run(run_workload())


async def run_workload():
    """Run workload"""
    # Set up workload information
    input_csv = conf_dict.experiment_info.get("trace_input")
    csv_dir = f"workload_generator/wits/{input_csv}"
    logger.info(f"Trace Information: {input_csv}\n")

    # Update running time for instances
    logger.info("Updating running time before starting workload")
    for inst_id, INST_INFO in whole_instance_info.items():
        INST_INFO.running_time = datetime.datetime.now()

    # Initialize job list and retry client
    jobs = []
    with open(csv_dir) as csv_file:
        reader = csv.reader(csv_file)
        retry_options = ExponentialRetry()
        async with RetryClient(retry_options=retry_options) as sess:
            idx = 0
            for each_row in reader:
                idx += 1
                arrival_time = idx
                try:
                    num_tasks = int(each_row[0])
                except IndexError:
                    continue
                # Send parallel requests
                for i in range(num_tasks):
                    duration_info["total_requests"] += 1
                    jobs.append(
                        asyncio.create_task(
                            send_request(
                                sess,
                                i,
                                arrival_time,
                                num_tasks,
                                duration_info["total_requests"],
                            )
                        )
                    )
                await asyncio.sleep(1)
        # Wait for all requests to finish
        await asyncio.gather(*jobs)

    logger.info("Ended Workload\n")


def get_number_of_requests_during_1_minute():
    """Returns the number of requests during the last minute"""

    num_requests_last_minute = (
            duration_info["total_requests"]
            - duration_info["accumulated_request_list_every_minute"][-1]
    )

    duration_info["accumulated_request_list_every_minute"].append(
        num_requests_last_minute
    )

    return num_requests_last_minute


def remove_workers(
        three_mins_elapsed_instances_to_terminate: List[InstanceInfoWithCPUClass],
):
    """Remove number of workers given in num_of_servers_to_terminate"""

    # List instance id list
    instances_to_terminate = [
        i.inst_id for i in three_mins_elapsed_instances_to_terminate
    ]

    logger.info(f"\t\tinstances_to_terminate -> {instances_to_terminate}")

    num_of_servers_to_terminate = len(three_mins_elapsed_instances_to_terminate)

    # Number of servers to remove should be less than existing servers
    if num_of_servers_to_terminate >= len(whole_instance_info):
        logger.info(
            f"\t\tWhole instance <= instances to terminate - Leaving only 1 worker"
        )
        instances_to_terminate = list(whole_instance_info.keys())[1:]
        logger.debug(f"\t\tinstances_to_terminate -> {instances_to_terminate}")

    remove_worker_and_terminate_instance(instances_to_terminate)

    return


def remove_worker_and_terminate_instance(instances_to_terminate):
    # Thread keepers
    lb_detachers_threads = []

    for each_inst_id in instances_to_terminate:
        # For Consistency
        instance_workers_lock.acquire()

        # Remove inst_id from whole_instance_info
        logger.info(f"\t\tRemoving {each_inst_id} from whole_instance_info")
        info_class: InstanceInfoClass = whole_instance_info.pop(each_inst_id)
        logger.info(
            f"\t\t\tCurrent ec2_id list from whole_instance_info ({len(whole_instance_info)})-"
            f" {whole_instance_info.keys()}"
        )

        terminated_instance_info[each_inst_id] = TerminatedInstanceInfoClass(
            info_class.vm_class_info,
            info_class.running_time,
            info_class.recent_instance,
            info_class.number_of_elapsed_minutes,
            info_class.cpu_util,
            termination_time=datetime.datetime.now(),
        )
        for idx, (inst_id, info) in enumerate(terminated_instance_info.items()):
            logger.debug(
                f"\t\t {idx + 1}/{len(terminated_instance_info)} terminated_inst {inst_id} - {info}"
            )

        # For Consistency
        instance_workers_lock.release()

        # Update public ip in Lambda & Terminate & Make servers unavailable in LoadBalancer
        lb_detacher = LBDetacher(info_class.vm_class_info)
        lb_detachers_threads.append(lb_detacher)

    # Parallel execution
    logger.info("\tStart Servers Unavailable & Termination")
    for thread in lb_detachers_threads:
        thread.start()

    # Wait for threads to finish
    for thread in lb_detachers_threads:
        thread.join()

    logger.info("\tFinished Servers Unavailable & Termination")

    # Update lambda env for mongodb after termination of all instances
    update_ec2_private_ip_env_in_lambda(lambda_arn_list_for_mongodb)


def determine_num_of_excess_servers(num_of_requests, num_of_vm, max_reqs, duration):
    """Return num of excess servers -> could be negative or positive"""

    # Initial and Exception Case
    if num_of_requests == 0:
        logger.info("\t\tSkipping since number of requests is 0")
        return 0

    experiment_case = conf_dict.experiment_info.get("experiment_case")

    if experiment_case in ["all_lambda", "test_lambda", "cold_start"]:
        logger.info("\t\t\tSkipping scaling decision for Lambda experiments")
        return 0

    logger.info("\t\t\tDetermining excess servers to launch/terminate")

    # Since there is a time for launching instances we have to calculate the gap
    if experiment_case in ["all_vm", "overprovision"]:

        # Add partial requests for newly launched instances + Add requests for remaining numbers
        max_reqs_per_minute = get_maximum_requests_for_vms(
            max_reqs, duration, num_of_vm
        )
        logger.info(
            f"\t\t\tmax_reqs_with_{int(num_of_vm)}_servers : {max_reqs_per_minute}"
        )

        # Get total excess - Subtract max nums to current number of requests
        total_excess_requests = num_of_requests - max_reqs_per_minute
        logger.info(f"\t\t\tTotal_excess is {total_excess_requests}")

        # OVER_PROVISION -> 2x resources
        if experiment_case == "overprovision":

            logger.info("\t\t\t[SCALING POLICY] - OVERPROVISION")

            # Number of servers to use -> could be positive or negative number
            num_of_servers_to_make = math.ceil(
                total_excess_requests / (max_reqs * duration)
            )

            if num_of_servers_to_make >= 0:
                num_of_servers_to_launch = 2 * num_of_servers_to_make
                logger.info(f"\t\t\tExcess servers: {num_of_servers_to_launch}")
                return num_of_servers_to_launch

            # If removing workers or no provision necessary
            else:
                num_of_servers_to_launch = num_of_servers_to_make
                logger.info(
                    f"\t\t\tServers to terminate: {abs(num_of_servers_to_launch)}"
                )
                return num_of_servers_to_launch

        # NO_OVER_PROVISION -> 1x resources
        else:
            logger.info("\t\t\t[SCALING POLICY] - NO OVERPROVISION ")
            num_of_servers_to_launch = math.ceil(
                total_excess_requests / (max_reqs * duration)
            )
            logger.info(f"\t\t\tExcess servers: {num_of_servers_to_launch}")
            return num_of_servers_to_launch

    # Assume newly launched VM can serve as if it is already running from the start (using Lambda)
    elif experiment_case == "microblend":

        logger.info(
            "\t\t\t use_case_for_experiment is MicroBlend -> recent launched VM can serve from beginning"
        )

        # Maximum Number of Requests for a minute which is num of servers * duration
        max_reqs_per_minute = num_of_vm * max_reqs * duration
        logger.info(
            f"\t\t\tmax_reqs_with_{int(num_of_vm)}_servers : {max_reqs_per_minute}"
        )

        # Subtract max_reqs_per_minute to current number of requests
        total_excess_requests = num_of_requests - max_reqs_per_minute
        logger.info(f"\t\t\tTotal_excess is {total_excess_requests}")

        # Divide with arrival rate to get number of servers====
        logger.info("\t[SCALING POLICY] - MicroBlend")
        num_of_servers_to_launch = math.ceil(
            total_excess_requests / (max_reqs * duration)
        )

        return num_of_servers_to_launch


def get_maximum_requests_for_vms(arrival_rate, duration, num_of_vm):
    """Depending on the experiment type, add partial requests for previously launched instances
    1. Iterate and find previously launched instances
    2. Calculate Gap and get partial requests depending on the gap
    """

    logger.info(
        "\t\t\tGet Maximum Requests including partial requests for previously launched instances"
    )

    recently_launched_instances = [
        info
        for info in whole_instance_info.values()
        if info.recent_instance
           and (datetime.datetime.now() - info.running_time).total_seconds() <= 57
    ]

    req_for_recent_instances = arrival_rate * sum(
        (datetime.datetime.now() - info.running_time).total_seconds()
        for info in recently_launched_instances
    )

    req_except_recent_instances = (
            (num_of_vm - len(recently_launched_instances)) * arrival_rate * duration
    )

    max_reqs_per_minute = req_for_recent_instances + req_except_recent_instances

    logger.info(f"\t\t\ttotal request for recent instances: {req_for_recent_instances}")
    logger.info(
        f"\t\t\ttotal request for non-recent instances: {req_except_recent_instances}"
    )
    logger.info(f"\t\t\tmax requests per minute: {max_reqs_per_minute}")

    return max_reqs_per_minute


def scale_down_resources_based_on_lowest_cpu_util(num_of_excess_server) -> None:
    """Scale down resources based on lowest cpu utilization"""

    # Number of servers to scale in  -> number of excess servers
    num_of_servers_to_scale_in = abs(num_of_excess_server)
    logger.info(f"\tRemove {num_of_servers_to_scale_in} server")

    # Set previous instance to False because at this time we didn't provision any instances
    logger.info(f"\tDoes not provision -> Setting previous instance's to False")
    for _, instance_information in whole_instance_info.items():
        instance_information.recent_instance = False

    # Insanity Check
    print_whole_instance_info()

    # Update and bring instances with sorted cpu utilization
    logger.info("\tScale in - choose the instances depending on lowest cpu util")
    instances_to_scale_in: List[InstanceInfoWithCPUClass] = get_instances_to_scale_in(
        whole_instance_info, lowest=True
    )

    # Show Instances Candidate
    try:
        logger.info(
            f"\t\tInstance Candidates to Scale In ({len(instances_to_scale_in)}): {instances_to_scale_in}"
        )
    except ValueError:
        logger.exception("Error with showing instances to scale in")

    logger.info(f"\t\tChoose the first {num_of_servers_to_scale_in} instances")
    instances_to_terminate = instances_to_scale_in[:num_of_servers_to_scale_in]
    logger.info(f"\t\tinstances to terminate is {instances_to_terminate}")

    # Remove workers
    t_for_removing_resource = threading.Thread(
        target=remove_workers, args=[instances_to_terminate]
    )
    t_for_removing_resource.start()


def terminate_idle_resources() -> None:
    """Set previous instance to False because at this time we didn't provision any instances"""
    logger.info(f"\t\tDoes not provision -> Setting previous instance's to False")
    for _, instance_information in whole_instance_info.items():
        instance_information.recent_instance = False

    # Insanity Check
    print_whole_instance_info()

    # Update and Bring instances with cpu utilization less than 10 percent
    logger.info("\tDecide if we need to scale in depending on cpu util")
    instances_to_scale_in: List[InstanceInfoWithCPUClass] = get_instances_to_scale_in(
        whole_instance_info
    )

    # Show Instances Candidate
    try:
        logger.info(f"\t\tInstances to Scale In : {instances_to_scale_in}")
    except ValueError:
        logger.exception("Exception Happened")

    # If there are instances to scale-in
    if instances_to_scale_in:
        logger.info("\tThere exist idle instances -> Scale in")
        num_of_servers_to_scale_in = len(instances_to_scale_in)
        logger.info(f"\tRemove {num_of_servers_to_scale_in} servers")
        # Remove workers
        t_for_removing_resource = threading.Thread(
            target=remove_workers, args=[instances_to_scale_in]
        )
        t_for_removing_resource.start()
    else:
        logger.info("\tNo idle resources to remove")


def scale_up_resources(num_of_excess_server):
    """Scale up & ways are vm (traditional) hybrid (MicroBlend)"""

    # Set previous instances status False since we will be spawning new resources
    logger.info(
        f"\t\t\tSet previous instances to False because we are launching new one"
    )
    for _, each_instance_info in whole_instance_info.items():
        each_instance_info.recent_instance = False

    # Case 1 ALL VM (Traditional)
    if provisioning_method == "vm":
        #  Bring new resources and add instances to whole_instance_info
        t = threading.Thread(
            target=launch_ec2_and_add_to_loadbalancer, args=[num_of_excess_server]
        )
        t.start()
        t.join()

    # Case 2 Use Lambda when provisioning (MicroBlend)
    if provisioning_method == "microblend":
        """
        This is the point where the compiler will generate hybrid code and coordinator code 
        that triggers the Lambda-based microservice. This code is then distributed across servers
        via AWS S3. During autoscaling, Loadcat will direct requests to the coordinator-code 
        which interacts with Lambda-based microservices.
        """
        # Start a thread to launch EC2 and add to load balancer
        ec2_thread = threading.Thread(target=launch_ec2_and_add_to_loadbalancer, args=[num_of_excess_server])
        ec2_thread.start()
    
        # Start the compiler thread and wait for it to finish
        # compiler_thread = threading.Thread(target=compiler.process_compiler, args=[original_code, source_file_name])
        # compiler_thread.start()
        # compiler_thread.join()  # Wait for the compiler thread to finish
    
        # Switch to using Lambda after launching the EC2 instances
        change_service_type_to_lambda()
    
        # Once EC2 and load balancer tasks are complete, switch back to using VMs
        ec2_thread.join()
        change_service_type_to_vm()

    logger.debug(
        f"\tCurrent Instances Info ({len(whole_instance_info)}) instances: {pformat(whole_instance_info)}"
    )

    # Insanity Check
    # print_whole_instance_info()


def increment_all_instances_number_of_elapsed_mins_by_one():
    """Increment elapsed time by one -> needed when terminating instances"""
    each_inst: InstanceInfoClass
    for inst_id, each_inst in whole_instance_info.items():
        each_inst.number_of_elapsed_minutes += 1


#
#
def start_policy():
    """Autoscale decision every duration"""

    logger.info("Start Scaling Policy")

    # Skip when use_case is for test
    experiment_case = conf_dict.experiment_info.get("experiment_case")
    logger.debug(f"\tExperiment Case: {experiment_case}")

    policy_duration = int(conf_dict.experiment_info.get("policy_duration"))

    if experiment_case in ["all_lambda", "test_lambda", "cold_start"]:
        logger.info("\t ALL_LAMBDA, TEST -> Skipping Scaling Policy")
        return

    else:
        start_time = time.time()
        while not index_for_stopping_scaling_policy:
            elapsed_time = time.time() - start_time
            sleep_time = policy_duration - (elapsed_time % policy_duration)
            logger.debug(f"\twake up after {sleep_time:.2f}")

            # Make autoscaling decision & Run in background
            autoscaling_task = threading.Thread(target=make_autoscaling_decision_and_provision)
            autoscaling_task.start()

    return


def make_autoscaling_decision_and_provision():
    """Decide number of servers to spawn/terminate & remove workers from LoadBalancer"""

    while not index_for_stopping_scaling_policy:

        logger.info("Make Autoscaling Decision")

        logger.info(f"\tIncrement number of elapsed time by 1")
        increment_all_instances_number_of_elapsed_mins_by_one()
        get_statistics()

        logger.info(f"\tGet number of arrival rate for duration")
        number_of_requests = get_number_of_requests_during_1_minute()
        logger.info(
            f"\tNumber of Total Requests for previous 1 minute is {number_of_requests}"
        )

        logger.info("\tDetermine number of excess servers")
        num_of_vm = len(whole_instance_info)
        arrival_rate = int(conf_dict.experiment_info["max_reqs_per_minute_per_vm"])
        policy_duration = int(conf_dict.experiment_info["policy_duration"])

        num_of_excess_server = determine_num_of_excess_servers(
            number_of_requests, num_of_vm, arrival_rate, policy_duration
        )
        logger.debug(f"\t\tNum of excess server is {num_of_excess_server}")

        num_of_excess_server = 0  # for test

        # Need to scale up
        if num_of_excess_server > 0:
            scale_up_resources(num_of_excess_server)

        # No instances to remove -> terminate idle instances
        elif num_of_excess_server == 0:
            terminate_idle_resources()

        # Terminate instances with the lowest cpu utilization
        elif num_of_excess_server < 0:
            scale_down_resources_based_on_lowest_cpu_util(num_of_excess_server)

        logger.info(
            f"\tEnd of Scaling Policy (But function inside scaling policy could be running)."
        )

        return


def copy_result_log():
    logger.info("Copying result")

    log_folder = conf_dict.experiment_info.get("result_dir")
    result_file_format = "experiment_result"
    log_list = glob.glob(f"{log_folder}/{result_file_format}_*.log")
    number_of_files_in_log = len(log_list)

    logger.info(f"Number of log files: {number_of_files_in_log}")

    if result_log_file_name:
        log_to_copy = f"{log_folder}/{result_file_format}_{number_of_files_in_log + 1}_{result_log_file_name}.log"
    else:
        try:
            log_to_copy = (
                f"{log_folder}/{result_file_format}_{number_of_files_in_log + 1}.log"
            )
        except NameError as e:
            logger.info(f"Exception {e} happened")
            log_to_copy = f"{log_folder}/{result_file_format}_{len(log_list) + 2}.log"

    logger.info(f"Writing to {log_to_copy}")
    copyfile(result_log_with_dir, log_to_copy)


def change_service_type_to_vm():
    """Change service type to VM"""
    global request_type

    request_type = "vm"
    logger.info("Run on VM")


def change_service_type_to_lambda():
    """Change service type to lambda"""
    global request_type

    request_type = "lambda"
    logger.info("Run on Lambda")


def get_average_cpu_for_ec2_instance(client, ec2_id, index):
    """Get average CPU utilization for EC2 instance"""
    try:
        response = client.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[{"Name": "InstanceId", "Value": ec2_id}],
            StartTime=datetime.datetime.now(datetime.timezone.utc)
                      - datetime.timedelta(seconds=180),
            EndTime=datetime.datetime.now(datetime.timezone.utc),
            Period=30,
            Statistics=["Average"],
        )
    except ValueError:
        logger.exception(f"\t\t\t\t\texception happened")
        return 0

    # Get list of CPU utilization every minute
    cpu_list_per_min = [x.get("Average") for x in response["Datapoints"]]
    logger.info(
        f"\t\t\tcpu_list_per_min of {ec2_id} ({index + 1}) is {cpu_list_per_min}"
    )

    try:
        # Get average of CPU utilization
        average_cpu_per_instance = sum(cpu_list_per_min) / len(cpu_list_per_min)

    except ZeroDivisionError as e:
        logger.info(f"\t\t\tException {e} -> setting CPU utilization to 0 ")
        average_cpu_per_instance = 0

    return average_cpu_per_instance


def get_instances_to_scale_in(
        whole_inst_info: dict, lowest=False
) -> List[InstanceInfoWithCPUClass]:
    """get cpu utilization for 3 minutes for all instances"""

    global whole_instance_info, cloudwatch_client

    instances_to_scale_in = []

    for i, (inst_id, inst_information) in enumerate(whole_inst_info.items()):
        number_of_elapsed_minutes = inst_information.number_of_elapsed_minutes

        if number_of_elapsed_minutes < 3:
            logger.info(
                f"\t\t{i + 1}th Instance {inst_id}'s elapsed time is less than 3 "
            )
            continue

        cpu_utilization = get_average_cpu_for_ec2_instance(
            cloudwatch_client, inst_id, i
        )

        inst_information.cpu_util = cpu_utilization

        if lowest:
            instances_to_scale_in.append(
                InstanceInfoWithCPUClass(inst_id, cpu_utilization)
            )
        elif cpu_utilization <= 10:
            logger.info(f"\t\t{i + 1}th Instance {inst_id}'s cpu util is less than 10")
            instances_to_scale_in.append(
                InstanceInfoWithCPUClass(inst_id, cpu_utilization)
            )

    if lowest:
        logger.info("\t\tSort instances from lowest cpu to highest cpu")
        instances_to_scale_in.sort(key=lambda x: x.cpu_util)

    return instances_to_scale_in


def fetch_instance_info():
    """
    Fetch whole_instance_info, worker_list_in_lbs, instance_list_in_lbs, and ec2 private ip address from pickle
    """
    pickle_file_dir = os.path.join(
        conf_dict.experiment_info.get("pickle_dir"), "instance_info.pickle"
    )

    with open(pickle_file_dir, "rb") as f:
        data = pickle.load(f)

    return (
        data["whole_instance_info"],
        data["instance_list_in_lbs"],
        data["worker_list_in_lbs"],
        data["ec2_private_ip"],
        data["instance_to_worker"],
    )


def save_instance_info_to_pickle(data_to_save: dict):
    """
    Save whole_instance_info, worker_list_in_lbs, instance_list_in_lbs in pickle
    """
    logger.info("Saving instance info to pickle")

    pickle_folder_dir = conf_dict.experiment_info.get("pickle_dir")
    pickle_file_dir = os.path.join(pickle_folder_dir, "instance_info.pickle")

    with open(pickle_file_dir, "wb") as f:
        pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

    logger.info(f"Pickle file saved at {pickle_file_dir}")


def print_whole_instance_info():
    """
    Print number of instances and details
    """
    logger.debug("\tPrint whole instance info")

    for idx, (key, instance_info) in enumerate(whole_instance_info.items()):
        logger.debug(f"\t\t{idx + 1}th instance - {key} - {instance_info}")

    logger.info("\n")


def get_statistics(show_all_durations=False):
    """
    Show instance info and SLO
    """
    logger.info("Getting statistics")

    print_whole_instance_info()

    for idx, (key, instance_info) in enumerate(terminated_instance_info.items()):
        logger.debug(f"\t{idx + 1}th instance - {key} - {instance_info}")

    if show_all_durations:
        show_statistics()


def show_statistics():
    """
    Show statistics
    """
    logger.debug(f"\tViolated durations: {duration_info['violated_duration_list']}")
    logger.debug(f"\tDuration list: {duration_info['duration_list']}")
    logger.debug(f"\tLambda duration list: {duration_info['lambda_duration_list']}")
    logger.debug(f"\tNumber of violations: {duration_info['number_of_violation']}")
    logger.debug(f"\tNumber of durations: {len(duration_info['duration_list'])}")
    logger.debug(f"\tTotal duration in seconds: {duration_info['duration_in_seconds']}")

    try:
        show_detailed_statistics()
    except ZeroDivisionError:
        logger.debug("\tZero division error\n")


def show_detailed_statistics():
    violations_ratio = (
                               1 - (duration_info["number_of_violation"] / len(duration_info["duration_list"]))
                       ) * 100
    logger.debug(f"\tSLO: {violations_ratio}")
    logger.debug(
        f"\tMedian of duration list: {np.median(duration_info['duration_list'])}"
    )

    if duration_info["duration_list"]:
        logger.debug(f"\tMaximum duration: {max(duration_info['duration_list'])}")

    avg_rps = len(duration_info["duration_list"]) / duration_info["duration_in_seconds"]
    logger.debug(f"\tAverage requests per second: {avg_rps}")
    logger.debug(f"\tMean of duration list: {np.mean(duration_info['duration_list'])}")
    logger.debug(
        f"\tRequests every minute: {duration_info['accumulated_request_list_every_minute']}"
    )

    if duration_info["lambda_duration_list"]:
        logger.debug(
            f"\tNumber of violations from lambda: {duration_info['number_of_violation_from_lambda']}"
        )
        logger.debug(
            f"\tNumber of lambda duration list: {len(duration_info['lambda_duration_list'])}"
        )
        logger.debug(
            f"\tMedian of lambda duration list: {np.median(duration_info['lambda_duration_list'])}"
        )
        logger.debug(
            f"\tMaximum duration of lambda duration list: {max(duration_info['lambda_duration_list'])}"
        )
        logger.debug(
            f"\tMean of lambda duration list: {np.mean(duration_info['lambda_duration_list'])}"
        )


if __name__ == "__main__":
    """Phase 1. Initial Experiment Setup - Get initial experiment setup, make new resources"""

    disable_workers_in_loadcat()
    terminate_previous_instances()
    launch_ec2_and_add_to_loadbalancer(
        conf_dict.server_config.get("init_number_of_instances"), init_phase=True
    )
    save_instance_info_to_pickle(data_to_save_in_pickle)

    # sys.exit(getframeinfo(currentframe ())) # Stop here for Phase 2 & 3

    """Phase 2 - Fetch Initial Instance Info"""

    # Fetch instance info from pickle file and assign it to variables
    (
        whole_instance_info,
        instance_id_list,
        worker_id_list,
        ec2_private_ip,
        instance_to_worker,
    ) = fetch_instance_info()

    """Phase 3 -> Workload and Policy"""

    workload_task = threading.Thread(target=start_workload)
    policy_task = threading.Thread(target=start_policy)
    workload_task.start()
    policy_task.start()
    workload_task.join()
    index_for_stopping_scaling_policy = True
    policy_task.join()

    copy_result_log()

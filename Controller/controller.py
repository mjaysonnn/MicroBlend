"""
Run workload and scaling policy with compiler

SYNOPSIS
========
::
    python3 controller.py

DESCRIPTION
===========
1. First find user pragma in function
2. Open Initial Resources
3. Start workload and scaling policy
4. When provisioning, use compiler to make hybrid case if necessary

ENVIRONMENT
==========
Do configuration before running workload

    USE_CASE_FOR_EXPERIMENTS

    LoadBalancer Configuration
        AMI_ID
        INSTANCE_WORKERS
        WORKLOAD_CHOICE
        NUMBER_OF_INSTANCES
        CPU IDLE PERCENT
        use_case_for_experiments

    Module Configuration



FILES
=====
1. Read csv in workload_folder
2. Choose benchmark in benchmark folder

"""

import asyncio
import csv
import datetime
import glob
import json
import logging.handlers
import math
import os
import pickle
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from inspect import currentframe, getframeinfo
from pprint import pformat
from shutil import copyfile
from typing import List, Optional

import aiohttp
import boto3
import numpy as np
import paramiko
import pytz
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# from Compiler import compiler_simplified


"""
Server Configuration 
"""

INSTANCE_TYPE = "c5.large"  # TODO : Change This
# Instance Info
NUMBER_OF_INSTANCES: int = 33  # TODO : Change This

# LoadBalancer AMI
BALANCER_EC2_IP: str = "3.239.118.123"  # TODO : Change This
BALANCER_ID: str = "61b971a8d98b43110b163415"  # TODO : Change This

"""
INPUT CSV 
"""

# INPUT_CSV = "wuts.csv"  # TODO: Change This
# INPUT_CSV = "stable_30_seconds.csv"
INPUT_CSV = "wits_average_130_factor_3_minute_60.csv"

INPUT_FOLDER = "../WorkloadGenerator/WITS/"  # TODO: Change This

MAX_REQUESTS_PER_SEC_PER_VM = 4  # TODO: Change This

SLO_CRITERIA = 1000  # TODO: Change This

# Server AMI
AMI_ID = "ami-06a850866841c1c9f"  # TODO : Change This

# Inference Types ======= # TODO: Change Workload
WORKLOAD_CHOICE = {
    "resnet18": "/resnet18/",
    "resnet152": "/resnet152/",
    "densenet": "/densenet/",
    "feature_generate": "/feat_gen/",
    "mat_mul": "/matmul/",
}

# User script for launching server AMI ===== # TODO: Change This
USER_DATA_SCRIPT = """#!/bin/bash 
    cd /home/ec2-user/MicroBlendServer
    pip3 install pandas    
    cd mat_mul
    uvicorn fastapi_server_matmul:app --reload --host=0.0.0.0 --port=5000 &
    """

# CONFIGURATION
USE_CASE_FOR_EXPERIMENT = "MicroBlend"  # TODO : Change This

"""
USE_CASE_FOR_EXPERIMENTS & WORKLOAD CONFIGURATION
"""

USE_CASE_FOR_EXPERIMENTS = ["ALL_VM", "OVERPROVISION",
                            "COLD_START", "ALL_LAMBDA", "MICROBLEND",
                            "TEST_FOR_1_MINUTE_VM", "TEST_FOR_1_MINUTE_LAMBDA"]

requests.adapters.DEFAULT_RETRIES = 40

NUMBER_OF_VIOLATION = 0
VIOLATED_DURATIONS = []

DURATION_LIST = []
TOTAL_DURATION = 0

REQUEST_TYPE = "vm"
PROVISIONING_METHOD = "ONLY_VM"
OVER_PROVISION = False

# Index for stopping policy
INDEX_FOR_STOPPING_SCALING_POLICY = False

# Termination Policy - CPU IDLE PERCENT =====
CPU_IDLE_PERCENT = 10

if USE_CASE_FOR_EXPERIMENT in USE_CASE_FOR_EXPERIMENTS:
    if USE_CASE_FOR_EXPERIMENT == "COLD_START":
        INPUT_CSV = "cold_start.csv"
        REQUEST_TYPE = "lambda"

        # PROVISIONING_METHOD = "ONLY_VM"
        # OVER_PROVISION = False
        INDEX_FOR_STOPPING_SCALING_POLICY = True

    elif USE_CASE_FOR_EXPERIMENT == "ALL_VM":
        REQUEST_TYPE = "vm"
        PROVISIONING_METHOD = "ONLY_VM"
        OVER_PROVISION = False

    elif USE_CASE_FOR_EXPERIMENT == "OVERPROVISION":
        REQUEST_TYPE = "vm"
        PROVISIONING_METHOD = "ONLY_VM"
        OVER_PROVISION = True

    elif USE_CASE_FOR_EXPERIMENT == "ALL_LAMBDA":
        REQUEST_TYPE = "lambda"
        INDEX_FOR_STOPPING_SCALING_POLICY = True

    elif USE_CASE_FOR_EXPERIMENT == "MICROBLEND":
        REQUEST_TYPE = "vm"
        PROVISIONING_METHOD = "Hybrid"
        OVER_PROVISION = False

    elif USE_CASE_FOR_EXPERIMENT == "TEST_FOR_1_MINUTE_LAMBDA":
        INPUT_CSV = "stable_1_minutes.csv"
        REQUEST_TYPE = "lambda"
        PROVISIONING_METHOD = "HYBRID"
        OVER_PROVISION = False
        INDEX_FOR_STOPPING_SCALING_POLICY = True

    elif USE_CASE_FOR_EXPERIMENT == "TEST_FOR_1_MINUTE_VM":
        INPUT_CSV = "stable_1_minutes.csv"
        REQUEST_TYPE = "vm"
        PROVISIONING_METHOD = "ONLY_VM"
        OVER_PROVISION = False
        INDEX_FOR_STOPPING_SCALING_POLICY = True
else:
    print("Wrong use case input")
    sys.exit(1)

"""
NGINX LoadBalancer Configuration
"""

SECURITY_GROUP = "sg-0ee3256ff0154c3d0"
KEY_NAME = "mjay_m1"

# INPUT CSV WITH DIRECTORY =======
INPUT_CSV_WITH_DIR = os.path.join(INPUT_FOLDER, INPUT_CSV)

# RESULT LOG NAMING =====

RESULT_LOG_NAME = None

if REQUEST_TYPE == "lambda":
    RESULT_LOG_NAME = f"{INPUT_CSV.split('.')[0]}_{REQUEST_TYPE}"

elif REQUEST_TYPE == "vm":
    if PROVISIONING_METHOD == "Hybrid":
        RESULT_LOG_NAME = (
            f"{INPUT_CSV.split('.')[0]}_microblend"
        )

    elif OVER_PROVISION:
        RESULT_LOG_NAME = (
            f"{INPUT_CSV.split('.')[0]}_"
            f"{REQUEST_TYPE}_"
            f"{PROVISIONING_METHOD.lower()}_over_provision"
        )
    else:
        RESULT_LOG_NAME = (
            f'{INPUT_CSV.split(".")[0]}_'
            f"{REQUEST_TYPE}"
            # f"_{PROVISIONING_METHOD.lower()}"
            f"_no_over_provision"
        )

# For saving instance workers =====
# INSTANCE_WORKERS = []
instance_workers_lock = threading.Lock()

# Save autoscaling index and running time =====

# Exclude installing for performance
# pip3 install -r requirements.txt


# AWS CREDENTIALS =======
CREDENTIALS = {}


@dataclass
class InstanceInfoWithCPUClass:
    inst_id: str
    cpu_util: float


@dataclass
class LoadBalancerConfigClass:
    # Instance Configuration
    instance_type: str = INSTANCE_TYPE
    init_number_of_instance: int = NUMBER_OF_INSTANCES
    security_groups: str = SECURITY_GROUP
    snid: str = "subnet-dd8dffd1"
    region_name: str = "us-east-1"
    snid_az: str = "us-east-1f"
    cost_per_hour: float = 0.096
    key_name: str = KEY_NAME
    max_requests_per_sec_per_vm: str = MAX_REQUESTS_PER_SEC_PER_VM

    # Server AMI ID
    image_id: str = AMI_ID

    # LoadBalancer Configuration
    balancer_ip_addr: str = BALANCER_EC2_IP
    balancer_id: str = BALANCER_ID
    balancer_ip_addr_with_port: str = balancer_ip_addr + ":26590"
    balancer_label: str = "LoadBalancer"
    basic_status_url: str = "/basic_status"

    # Workload Configuration
    input_csv: str = INPUT_CSV_WITH_DIR
    workload_scaling_factor: int = 1

    # Scaling Policy Configuration
    duration: int = 60

    # default_weight: int = 1
    # baseline: int = 80
    # avail_zone: str = ""
    # logfile: str = ""


@dataclass
class ModuleConfigClass:
    # File Name
    f_name: str = "resnet18_vm_for_test.py"

    # Workload Result Log
    result_logfile_folder: str = "../Log/Result"
    result_logfile_name: str = "../Log/Result/microblend_result.log"

    worker_log_folder: str = "../Log/Worker"
    worker_log_file: str = "../Log/Worker/workers.json"

    pickle_file: str = "../Log/Pickle/vm_instance.pkl"

    # Compiler Result Log
    bench_dir: str = "../BenchmarkApplication/resnet18"
    module_dir: str = "import_modules"
    output_path_dir: str = "output"
    lambda_code_dir_path: str = output_path_dir + "/lambda_codes"
    deployment_zip_dir: str = output_path_dir + "/deployment_zip_dir"
    hybrid_code_dir: str = output_path_dir + "/hybrid_vm"
    hybrid_code_file_name: str = "compiler_generated_hybrid_code"
    bucket_for_hybrid_code: str = "coco-hybrid-bucket-mj"
    bucket_for_lambda_handler_zip: str = "faas-code-deployment-bucket"
    # log_folder_dir: str = "coco-hybrid-bucket"


# Fetch Compiler module configuration
module_conf = ModuleConfigClass()

# Fetch Load Balancer Config
lb_conf = LoadBalancerConfigClass()
s_factor = lb_conf.workload_scaling_factor
lb_ip = lb_conf.balancer_ip_addr


def return_logger():
    # Logging Configuration
    modules_for_removing_debug = [
        "urllib3",
        "s3transfer",
        "boto3",
        "botocore",
        "urllib3",
        "requests",
        "paramiko",
    ]
    for name in modules_for_removing_debug:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    log_inst = logging.getLogger(__name__)
    log_format = logging.Formatter("%(asctime)s [%(levelname)-6s] [%(funcName)-58s:%(lineno)-4s]  %(" "message)s")
    # File Handler
    file_handler = logging.FileHandler(module_conf.result_logfile_name)
    file_handler.setFormatter(log_format)
    log_inst.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    log_inst.addHandler(console_handler)
    # SET LEVEL
    log_inst.setLevel(logging.DEBUG)

    return log_inst


# Not Using this for now
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


def get_initial_request() -> int:
    """Get starting number of requests
    :return: Starting number
    """
    logger.info("Get Initial Request")

    lb_ip_addr = lb_conf.balancer_ip_addr
    basic_status = lb_conf.basic_status_url
    url_to_request = "http://" + lb_ip_addr + basic_status

    with requests.Session() as s:
        try:
            number_res = s.get(url_to_request, timeout=5)
            res_text = number_res.text.split()

        except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
        ) as e:
            logger.info(f"Exception :{e}")
            logger.info("No Load Balancer running or Wrong IP Address")
            logger.info("Setting request to 0")
            return 0
        else:
            assert number_res.status_code == 200
            prev_request = int(res_text[res_text.index("Reading:") - 1])
            logger.info(f"\tstarting request is {prev_request}")
            return prev_request


def empty_result_log():
    """Empty microblend_result.log
    :return: None
    """
    logger.info("Empty microblend_result.log beforehand")
    open(module_conf.result_logfile_name, "w").close()
    assert os.path.getsize(module_conf.result_logfile_name) == 0


def make_all_workers_unavailable(initial_work_id_list: List[Optional[str]] = None):
    """
    Iterate through worker configuration and make each worker in lb unavailable
    """
    logger.info("Make previous workers unavailable")

    # If empty make a empty list
    if initial_work_id_list is None:
        initial_work_id_list = []

    # List worker json file list
    os.chdir("..")  # move to parent directory

    # Fetch all worker files -> will be used for removing except original files
    w_folder = "Log/Worker"
    w_file_format = "workers"
    original_log_file_name = "workers.json"
    worker_file_list = glob.glob(f"{w_folder}/{w_file_format}*.json")
    logger.info(f'\tWorker Files are {worker_file_list}')

    # Connect to LB and fetch worker_id
    html_page = urllib.request.urlopen(f"http://{lb_conf.balancer_ip_addr}:26590/balancers/{lb_conf.balancer_id}")
    soup = BeautifulSoup(html_page, "html.parser")
    worker_id_list_from_lb = []
    for link in soup.findAll('a'):
        if str(link.get('href')).startswith('/servers/'):
            split_href = str(link.get('href')).split("/")
            for i, c in enumerate(split_href):
                if c == "servers":

                    # logger.debug(split_href)
                    # Skip Initial Worker Id

                    if split_href[i + 1] not in initial_work_id_list:
                        worker_id_list_from_lb.append(split_href[i + 1])

    # logger.info(f"worker_file_list : {worker_id_list_from_lb}")
    # logger.info(f"len(worker_file_list) : {len(worker_id_list_from_lb)}")

    # Make dummy dictionary except initial worker id list
    worker_dict = defaultdict()
    for each_worker in worker_id_list_from_lb:
        data = defaultdict()
        worker_id = each_worker
        data["settings.address"] = ("0.0.0.0" + ":" + str(5000))
        data["settings.weight"] = 1
        data["settings.availability"] = "unavailable"
        data["settings.vmtype"] = "ondemand"
        data["label"] = "previous_one" + "_unused"
        worker_dict[worker_id] = data

    # worker_dict = make_worker_list_info(worker_file_list)

    # logger.debug(f"\t\t Worker Dict is {pformat(worker_dict)}")

    # Do request and make workers unavailable
    make_servers_unavailable(worker_dict),  # sys.exit(getframeinfo(currentframe()))

    # Remove worker log except worker.json
    for each_worker_json in worker_file_list:
        original_f_name = os.path.join(w_folder, original_log_file_name)

        if original_f_name in each_worker_json:  # Original File
            logger.info(f"\tEmpty and Skip original log")
            open(original_f_name, "w").close()
            assert os.path.getsize(original_f_name) == 0
            continue
        else:
            os.remove(each_worker_json)
            logger.debug(f"\t\tDeleted {each_worker_json}")

    os.chdir("Controller")
    # logger.debug(os.getcwd())
    logger.info("\tFinished making all workers unavailable")


def make_servers_unavailable(worker_info_dict):
    """Make all previous servers unavailable except initial servers

    :param worker_info_dict: worker id with label
    :return: None
    """
    if worker_info_dict is None:
        return

    logger.info("\tMake each server unavailable")

    exception_str = (
        "Inconsistency with loadbalancer configuration "
        "-> remove or overwrite json file"
    )

    # logger.debug(f"\t{worker_info_dict}")
    # logger.debug(f"\t{worker_ids}")

    if worker_info_dict:
        for each_worker_url, worker_data in worker_info_dict.items():

            # logger.debug(each_worker_url)
            # logger.debug(worker_data)

            if each_worker_url is None:
                logger.info("\tSkipping since each_worker_url is None")
                continue

            # if each_worker_url in worker_ids:
            #     logger.info(f"\tSkip Initial Worker - {each_worker_url}")
            #     continue

            else:
                post_request_to_make_server_unavailable(each_worker_url, exception_str, worker_data)


def post_request_to_make_server_unavailable(
        each_worker_url, exception_str, worker_data
):
    # logger.debug(each_worker_url)
    uri = "/servers/" + each_worker_url + "/edit"
    url = lb_conf.balancer_ip_addr_with_port + uri

    with requests.session() as sess:

        try:
            res = sess.post(url="http://" + url, timeout=3, data=worker_data)

        except TimeoutError as e:
            logger.info(f"\t Exception {e} -> due to previous one")

        except requests.exceptions.ConnectionError as error:
            logger.info(f"Exception -> {error}")
            logger.info(f"\t\t{exception_str}")

        except requests.exceptions.ReadTimeout:
            logger.info("\tMove on the next server")

        else:
            assert res.status_code == 200
            # logger.info(worker_data.get("label"))


def make_worker_list_info(workers_info_list):
    workers_dict_info = defaultdict()

    logger.info("\tFetch Worker Info from each worker log")

    for worker_file in workers_info_list:

        # logger.debug(worker_file)

        if os.stat(worker_file).st_size != 0:
            # logger.debug("open")

            with open(worker_file) as json_file:
                json_data = json.load(json_file)
                # logger.debug(json_data)
                make_unavailable_label_for_each_server(json_data, workers_dict_info)
    # logger.debug(workers_dict_info)

    return workers_dict_info


def make_unavailable_label_for_each_server(json_data, worker_info_dict):
    # logger.info("Make Unavailable Label for each worker")

    for _, instance_info_list in json_data.items():

        inst_info: dict
        for inst_info in instance_info_list:
            data = defaultdict()
            worker_id = inst_info.get('worker_id')
            data["settings.address"] = (inst_info.get("instance_ip") + ":" + str(5000))
            data["settings.weight"] = 1
            data["settings.availability"] = "unavailable"
            data["settings.vmtype"] = "ondemand"
            data["label"] = inst_info.get("instance_id") + "_unused"
            worker_info_dict[worker_id] = data


def terminate_instance(instance_id, region_name="us-east-1"):
    client = boto3.client("ec2", region_name=region_name, **CREDENTIALS)
    client.terminate_instances(InstanceIds=[instance_id])
    logger.info(f"\tTerminating instance - {instance_id}")


def terminate_previous_instances(inst_id_list=None):
    """Terminate instance before experiment -> emtpy worker_log_file
    """
    if inst_id_list is None:
        inst_id_list = []

    logger.info("Terminate previous instances")

    client = boto3.client("ec2", **CREDENTIALS, region_name="us-east-1")

    custom_filter = [
        {"Name": "tag:Name", "Values": ["ServerForApplication-MJ"]},
        {"Name": "instance-state-name", "Values": ["running"]},
    ]

    response = client.describe_instances(Filters=custom_filter)
    try:
        for res in response["Reservations"]:
            for i in res["Instances"]:
                each_inst_id = i["InstanceId"]

                if not inst_id_list:
                    terminate_instance(each_inst_id, "us-east-1")
                else:
                    if each_inst_id in inst_id_list:
                        logger.info(f"\tSkip Initial Instance - {each_inst_id}")
                        continue

        if not response["Reservations"]:
            raise ValueError

    except ValueError:
        logger.info('\tNo instances of "ServerApplication" running')

    finally:
        logger.info("\tFinished terminating instances beforehand")


def remove_contents_in_worker_log():
    logger.info("\tRemoving log file before experiment")
    try:
        open(module_conf.worker_log_file, "w").close()
        logger.info("\tRemoved contents in worker.json beforehand")

    except FileNotFoundError:
        logger.info(f"\t{module_conf.worker_log_file} Not Found ")


def create_instance(num, instance_type, image_id, security_g, snid, key_name):
    ec2 = boto3.resource("ec2", region_name="us-east-1", **CREDENTIALS)
    instance = ec2.create_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        KeyName=key_name,
        MaxCount=num,
        MinCount=1,
        Monitoring={"Enabled": True},
        # Placement={
        #     'AvailabilityZone': avail_zone,
        # },
        SecurityGroupIds=[security_g],
        SubnetId=snid,
        DisableApiTermination=False,
        InstanceInitiatedShutdownBehavior="stop",
        UserData=USER_DATA_SCRIPT,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": "ServerForApplication-MJ"}],
            },
        ],
    )

    return [instance[i].instance_id for i in range(len(instance))]


def update_worker_attribute(
        balancer_ip_addr_with_port,
        vm,
        worker_id,
        attr_dict: dict = None,
        new_resource=False,
        hybrid_code=False,
        offload_worker=False,
        get_back_to_original_worker=False,
        remove_worker=False,
):
    def get_uri_for_edit(w_id):
        uri = "/servers/" + w_id + "/edit"
        url = "http://" + balancer_ip_addr_with_port + uri
        return url

    if attr_dict is None:
        attr_dict = {}

    # When initialing resource -> vm : available, hybrid_code - backup
    if new_resource:
        address_for_editing = get_uri_for_edit(worker_id)
        info_for_server = attr_dict

        if hybrid_code:
            info_for_server["label"] = vm.get_instance_id() + "_5001"
        else:
            info_for_server["label"] = vm.get_instance_id() + "_5000"

        info_for_server["settings.vmtype"] = "ondemand"
        info_for_server["settings.weight"] = vm.get_weight()
        response = requests.post(url=address_for_editing, data=info_for_server)
        assert response.status_code == 200
        vm.add_data_to_data_per_flask_id(dict(info_for_server))

        # 0 is vm , 1 is hybrid
    elif offload_worker:

        address_for_editing = get_uri_for_edit(vm.return_worker_id_list()[1])  # address_for_editing for server with VM
        info_for_server = vm.get_data_per_flask_id()[1]
        info_for_server["settings.availability"] = ["available"]  # make vm unavailable
        response = requests.post(url=address_for_editing, data=info_for_server)
        assert response.status_code == 200

        address_for_editing = get_uri_for_edit(vm.return_worker_id_list()[0])  # address_for_editing for server with VM
        info_for_server = vm.get_data_per_flask_id()[0]
        info_for_server["settings.availability"] = ["unavailable"]  # make vm unavailable
        response = requests.post(url=address_for_editing, data=info_for_server)
        assert response.status_code == 200

    elif get_back_to_original_worker:
        address_for_editing = get_uri_for_edit(vm.return_worker_id_list()[0])  # address_for_editing for server with VM
        info_for_server = vm.get_data_per_flask_id()[0]
        info_for_server["settings.availability"] = ["available"]  # make vm unavailable
        response = requests.post(url=address_for_editing, data=info_for_server)
        assert response.status_code == 200

        address_for_editing = get_uri_for_edit(
            vm.return_worker_id_list()[1]
        )  # address_for_editing for server with VM
        info_for_server = vm.get_data_per_flask_id()[0]
        info_for_server["settings.availability"] = ["unavailable"]  # make vm unavailable
        response = requests.post(url=address_for_editing, data=info_for_server)
        assert response.status_code == 200

    elif remove_worker:
        address_for_editing = get_uri_for_edit(vm.worker_id)
        # address_for_editing = get_uri_for_edit(worker_id)

        # First Backup for finishing existing application
        info_for_server = vm.get_data_per_flask_id()[0]
        info_for_server["settings.availability"] = ["backup"]  # make vm
        response = requests.post(url=address_for_editing, data=info_for_server)
        assert response.status_code == 200

        # Give time to finish existing application
        time.sleep(5)

        # Unavailable
        info_for_server = vm.get_data_per_flask_id()[0]
        info_for_server["settings.availability"] = ["unavailable"]
        response = requests.post(url=address_for_editing, data=info_for_server)
        assert response.status_code == 200


class VM:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.instance_ip = self.get_instance_ip()  # 172.31.90.171
        self.worker_id = None
        self.vmtype = "ondemand"
        self.weight = 1
        self.worker_id_list = []
        self.data_per_worker_id = []

    def set_weight(self, weight):
        self.weight = weight

    def get_weight(self):
        return self.weight

    def get_instance_id(self):
        return self.instance_id

    def add_to_worker_id_list(self, worker_id):
        self.worker_id_list.append(worker_id)

    def return_worker_id_list(self):
        return self.worker_id_list

    def add_data_to_data_per_flask_id(self, data_dict):
        self.data_per_worker_id.append(data_dict)

    def get_data_per_flask_id(self):
        return self.data_per_worker_id

    def set_worker_id(self, worker_id):
        self.worker_id = worker_id

    def get_worker_id(self):
        return self.worker_id

    def set_instance_type(self, vmtype):
        self.vmtype = vmtype

    def get_instance_ip(self, ip_type="PrivateIpAddress"):
        """
        returns the public or private ip_address address of the instance.
        to get the public ip_address address, set ip_type to "PublicIp"
        """
        client = boto3.client("ec2", **CREDENTIALS, region_name="us-east-1")
        response = client.describe_instances(InstanceIds=[self.instance_id])
        return response["Reservations"][0]["Instances"][0]["NetworkInterfaces"][0][
            ip_type
        ]

    def get_instance_cpu_util(self):
        client = boto3.client("cloudwatch", **CREDENTIALS, region_name="us-east-1")
        logger.info(f"instance_id is : {self.instance_id}")
        response = client.get_metric_statistics(
            Namespace="AWS/EC2",
            MetricName="CPUUtilization",
            Dimensions=[{"Name": "InstanceId", "Value": self.instance_id}],
            # StartTime=datetime.datetime.now() - datetime.timedelta(minutes=15),
            StartTime=datetime.datetime.now() - datetime.timedelta(seconds=180),
            EndTime=datetime.datetime.now(),
            Period=30,
            # Statistics=["SampleCount", "Average", "Sum", "Minimum", "Maximum"],
            Statistics=["Average"],
        )
        # print(response['Datapoints']
        # logger.info(*response['Datapoints'], sep='\n')

        with open(module_conf.pickle_file, "wb") as output:  # FOR TEST
            pickle.dump(response["Datapoints"], output, pickle.HIGHEST_PROTOCOL)
        for x in response["Datapoints"]:
            logger.info(x)
        return response
        # return response["Datapoints"][0]["Average"]


@dataclass
class InstanceInfoClass:
    vm_class_info: VM = None
    running_time: datetime.datetime = datetime.datetime.now()
    previously_launched_instances: bool = False
    number_of_elapsed_minutes: int = 0
    cpu_util: float = 0


@dataclass
class TerminatedInstanceInfoClass(InstanceInfoClass):
    termination_time: datetime.datetime = datetime.datetime.now()


def update_weight_for_vm_and_hybrid(vm_class: VM, offload: bool):
    balancer_ip_addr_with_port = lb_conf.balancer_ip_addr_with_port

    if offload:  # Make VM Unavailable
        update_worker_attribute(
            balancer_ip_addr_with_port,
            vm_class,
            vm_class.worker_id,
            offload_worker=True,
        )

    else:  # Make VM Available
        update_worker_attribute(
            balancer_ip_addr_with_port,
            vm_class,
            vm_class.worker_id,
            get_back_to_original_worker=True,
        )


def utc_to_local(utc_dt):
    local_tz = pytz.timezone("Asia/Seoul")
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_tz.normalize(local_dt)  # .normalize might be unnecessary


def find_public_ip_and_launch_time(ec2_instance_id):
    client = boto3.client("ec2", region_name="us-east-1", **CREDENTIALS)

    instance_public_ip, ec2_instance_launch_time = None, None

    while True:
        try:
            response1 = client.describe_instances(InstanceIds=[ec2_instance_id])
            # logger.debug(response1)

            for reservation in response1["Reservations"]:
                # logger.debug(reservation)

                for instance in reservation["Instances"]:
                    # logger.debug(instance.get("InstanceId"))

                    # if instance.get("InstanceId") == ec2_instance_id:
                    instance_public_ip = instance.get("PublicIpAddress")
                    ec2_instance_launch_time = instance.get("LaunchTime")
                    # logger.debug(instance_public_ip)
                    if instance_public_ip is None:
                        time.sleep(1)
                        logger.info(f"\tRetrieving PublicIpAddress for {ec2_instance_id}")
                        raise TypeError

        except TypeError:
            # logger.debug("ip is none")
            continue

        break
        # logger.debug(instance.get("PublicIpAddress"))
        # logger.debug(instance.get("LaunchTime"))
        # logger.debug(instance.get("PublicIpAddress"))
        # instance_public_ip = instance.get("PublicIpAddress")

    # sys.exit()
    return instance_public_ip, utc_to_local(ec2_instance_launch_time).replace(tzinfo=None)


def check_ssh_for_spawning_time(ip_addr):
    try:
        get_session(ip_addr)
    except Exception as e:
        logger.info(f"\t\tSSH exception for ip_address: {ip_addr} - {e}")
        return False
    else:
        return datetime.datetime.now()


def get_session(ip_address):
    ssh = paramiko.client.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
    # key_name = "/Users/mj/.ssh/mjay.pem"
    key_name = f"/Users/mj/.ssh/{lb_conf.key_name}"
    logger.info(f"\tssk key is : {key_name}")
    ssh.connect(ip_address, username="ec2-user", key_filename=key_name)
    # sftp = ssh.open_sftp()
    return ssh


def wait_till_spawning_complete(inst_id):
    """
    Make sure all instances are connected by ssh success
    """
    global WHOLE_INSTANCE_INFO

    logger.info(f"\tinstance is {inst_id}")
    instance_public_address, instance_launch_time = find_public_ip_and_launch_time(inst_id)
    logger.info(f"\tpublic address for {inst_id} is {instance_public_address}")
    while True:
        ssh_success_time = check_ssh_for_spawning_time(ip_addr=instance_public_address)
        time.sleep(0.5)
        if ssh_success_time:
            logger.info(f"\t{inst_id} is running currently")

            logger.info("\tSave Running Time in WHOLE_INSTANCE_INFO -> Will update when done adding to LB")
            whole_instance_info = InstanceInfoClass(running_time=datetime.datetime.now())
            # inst_info: InstanceInfoClass = WHOLE_INSTANCE_INFO.get(inst_id)
            WHOLE_INSTANCE_INFO[inst_id] = whole_instance_info
            logger.info(f"\tWHOLE_INSTANCE_INFO ({len(WHOLE_INSTANCE_INFO)} -> {pformat(WHOLE_INSTANCE_INFO)}")
            # inst_info.running_time = datetime.datetime.now()

            break

    vm_info_of_instance = VM(instance_id=inst_id)
    vm_info_of_instance.set_weight(weight=1)

    # Add instances to INSTANCE_WORKERS
    # Consistency
    instance_workers_lock.acquire()

    # Append VM instance to global keeper
    logger.info("\tAdding Instance info class to WHOLE_INSTANCE_INFO")
    instance_info_class: InstanceInfoClass = WHOLE_INSTANCE_INFO.get(vm_info_of_instance.instance_id)
    instance_info_class.vm_class_info = vm_info_of_instance
    instance_info_class.previously_launched_instances = True
    instance_info_class.cpu_util = 0
    # instance_info_class = InstanceInfoClass(vm_info_of_instance, datetime.datetime.now(), True, 0)
    # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id] = instance_info_class
    # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].vm_class_info = vm_info_of_instance
    # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].number_of_elapsed_minutes = 0
    # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].running_time = datetime.datetime.now()
    # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].previously_launched_instances = True

    # Consistency
    instance_workers_lock.release()

    logger.info(f"\tStart adding instance {inst_id} to LB")
    thread_for_attacher = LBAttacher(vm=vm_info_of_instance)
    thread_for_attacher.start()
    thread_for_attacher.join()

    logger.info(f"\tUpdate {vm_info_of_instance.instance_id} running-time in WHOLE_INSTANCE_INFO")
    instance_info_class.running_time = datetime.datetime.now()
    # threads_for_adding_to_lb.append(thread_for_attacher)
    logger.info(f"\tCurrent Instances Info ({len(WHOLE_INSTANCE_INFO)}) instances:"
                f" {pformat(WHOLE_INSTANCE_INFO)}")

    return


class LBAttacher(threading.Thread):
    def __init__(self, vm):
        super(LBAttacher, self).__init__()
        self.vm = vm

    def run(self):
        # TODO: change this - Time for downloading image and other things before running mxnet
        time.sleep(3)
        add_worker_to_lb(lb_conf.balancer_id, lb_conf.balancer_ip_addr_with_port, self.vm, )


def turn_off_worker(lb_url, vm):
    return update_worker_attribute(
        lb_url, vm, {"settings.availability": "unavailable"}, remove_worker=True
    )


class LBDetacher(threading.Thread):
    def __init__(self, vm):
        super(LBDetacher, self).__init__()
        self.vm = vm

    def run(self):
        logger.info("\tStart Removing instance {} from LB: ".format(self.vm.get_instance_id()))
        turn_off_worker(lb_conf.balancer_ip_addr_with_port, self.vm)
        logger.info("\tFinished Removing instance {} from LB: ".format(self.vm.get_instance_id()))

        logger.info("\tStart terminating instance {}".format(self.vm.get_instance_id()))
        terminate_instance(self.vm.get_instance_id(), lb_conf.region_name)
        logger.info("\tFinished terminating instance {}".format(self.vm.get_instance_id()))


def add_worker_to_lb(balancer_id, lb_address, vm):
    # noinspection HttpUrlsUsage
    """add a new worker to load balancer.
    The attr_dict must contain backup and down attrs and both should be set to false
    weight attr also is in this info_for_server structure and default value is 1

    example) balancer_ip_address_with_port='http://100.26.175.206:26590'
    """
    # First create server with default settings
    # Port number
    port_number = 5000

    address_for_new_server = "/balancers/" + balancer_id + "/servers/new"

    full_address_for_new_server = "http://" + lb_address + address_for_new_server

    info_for_server = defaultdict()
    info_for_server["settings.address"] = vm.get_instance_ip() + ":" + str(port_number)
    info_for_server["label"] = vm.get_instance_id()

    # make request of making new server
    try:
        r = requests.post(url=full_address_for_new_server, data=info_for_server)
    except requests.exceptions.ConnectionError as e:
        logger.info(f"Exception {e} happened")
    else:
        worker_id = r.text.strip("\n")
        assert r.status_code == 200
        vm.set_worker_id(worker_id)

        # After creation, update server
        info_for_server["settings.weight"] = vm.get_weight()
        info_for_server["settings.availability"] = "available"
        update_worker_attribute(
            lb_address,
            vm,
            worker_id,
            attr_dict=info_for_server,
            new_resource=True,
            hybrid_code=False,
        )

    logger.info(f"\tFinish adding instance {vm.instance_id} to LB")

    # Write to json file
    logger.info("\tWriting worker info to json file")
    with open(module_conf.worker_log_file) as json_file:

        try:
            json_data = json.load(json_file)
        except ValueError as e:
            logger.info(f"\tException : {e} -> Json File is empty -> Filling a new one")
            json_data = defaultdict()
            json_data["workers"] = []

        json_data["workers"].append(vm.__dict__)

    with open(module_conf.worker_log_file, "w") as json_file:
        json.dump(json_data, json_file)
    # json_lock.release()
    # return r.text


def bring_new_resource(num_of_workers_to_add=0, init_phase=False) -> None:
    """
    Create Instance -> Wait for spawning ->
    Keep Track of regular instances -> Add to LoadBalancer
    """

    # Fetch global instances
    # global INSTANCE_WORKERS

    # Update newly instance list
    # global INITIAL_OR_LAUNCHED_1_MINUTE_AGO_INSTANCES

    global WHOLE_INSTANCE_INFO

    if init_phase:
        logger.info(f"\tRequested {lb_conf.init_number_of_instance} instances")
        logger.info(f"\tInstance Type: {lb_conf.instance_type}")
        # logger.info(f"\tInstance Number: {lb_conf.init_number_of_instance}")
        inst_id_list_to_launch = create_instance(
            num=lb_conf.init_number_of_instance,
            instance_type=lb_conf.instance_type,
            image_id=lb_conf.image_id,
            snid=lb_conf.snid,
            security_g=lb_conf.security_groups,
            key_name=lb_conf.key_name
        )

    else:  # Non-init_phase phase from scaling policy
        logger.info(f"\t\t\tRequested {num_of_workers_to_add} instances")
        inst_id_list_to_launch = create_instance(
            num=num_of_workers_to_add,
            instance_type=lb_conf.instance_type,
            image_id=lb_conf.image_id,
            snid=lb_conf.snid,
            security_g=lb_conf.security_groups,
            key_name=lb_conf.key_name
        )

    # Threads for waiting till all instances are running
    threads_for_waiting_for_spawning = []

    for instance_id in inst_id_list_to_launch:
        each_thread = threading.Thread(target=wait_till_spawning_complete, args=[instance_id])
        threads_for_waiting_for_spawning.append(each_thread)

    for job in threads_for_waiting_for_spawning:
        job.start()

    # for instance_id in inst_id_list_to_launch:
    #     # For each instance, make VM class that contains weight or other info
    #     vm_info_of_instance = VM(instance_id=instance_id)
    #     vm_info_of_instance.set_weight(weight=1)
    #
    #     # Add instances to INSTANCE_WORKERS
    #     # Consistency
    #     instance_workers_lock.acquire()
    #
    #     # Append VM instance to global keeper
    #     logger.info("\tAdding Instance info class to WHOLE_INSTANCE_INFO")
    #     instance_info_class: InstanceInfoClass = WHOLE_INSTANCE_INFO.get(vm_info_of_instance.instance_id)
    #     instance_info_class.vm_class_info = vm_info_of_instance
    #     instance_info_class.previously_launched_instances = True
    #     instance_info_class.cpu_util = 0
    #     # instance_info_class = InstanceInfoClass(vm_info_of_instance, datetime.datetime.now(), True, 0)
    #     # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id] = instance_info_class
    #     # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].vm_class_info = vm_info_of_instance
    #     # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].number_of_elapsed_minutes = 0
    #     # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].running_time = datetime.datetime.now()
    #     # WHOLE_INSTANCE_INFO[vm_info_of_instance.instance_id].previously_launched_instances = True
    #
    #     # Consistency
    #     instance_workers_lock.release()
    #
    #     logger.info(f"\tStart adding instance {instance_id} to LB")
    #     thread_for_attacher = LBAttacher(vm=vm_info_of_instance)
    #     thread_for_attacher.start()
    #     threads_for_adding_to_lb.append(thread_for_attacher)

    for job in threads_for_waiting_for_spawning:
        job.join()

    # for thread_for_attacher in threads_for_adding_to_lb:
    #     thread_for_attacher.join()

    logger.info(f"\tFinished adding all instances to LB")
    # logger.info(f"\tRunning Instances are {list(INSTANCE_DICT_INFO.keys())}")
    # logger.info(f"\tINSTANCE_DICT_INFO : {INSTANCE_DICT_INFO}")
    # logger.info(f"\tWHOLE_INSTANCE_INFO : {pformat(WHOLE_INSTANCE_INFO)}")


async def send_req(session, num=0, arrive_time=0, num_of_tasks=0, total_reqs=0, cold_s=False):
    # conn = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False,)
    global NUMBER_OF_VIOLATION
    global VIOLATED_DURATIONS
    global DURATION_LIST
    global TOTAL_DURATION

    TOTAL_DURATION = arrive_time

    # retry_client = RetryClient(session)

    # Cold Start Lambda Use Case
    if cold_s:
        logger.info("Invoke Cold Start")
        address_to_request = (
                "http://"
                + lb_conf.balancer_ip_addr
                + WORKLOAD_CHOICE.get("mat_mul")
                + "lambda"
        )

        retries = Retry(total=15, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        # async with aiohttp.ClientSession(connector=conn) as session:
        # async with aiohttp.ClientSession() as session:
        try:
            async with session.get(address_to_request, timeout=50) as resp:
                logger.info("Invoked Lambda for Cold Start")
                assert resp.status == 200

        except aiohttp.ClientConnectionError as e:
            logger.info(f"Error {e}")

    else:
        address_to_request = (
                "http://"
                + lb_conf.balancer_ip_addr
                + WORKLOAD_CHOICE.get("mat_mul")
                + REQUEST_TYPE
        )

        # TODO: when REQUEST_TYPE = lambda -> address_to_request = "compiler_created_lambda_arn"

        # logger.info(address_to_request)

        # logger.info(f"address_to_request is {address_to_request}")
        # For error with concurrency -> need a milli gap
        await asyncio.sleep(0.001)

        # Measure start time
        start = time.time()

        try:
            async with session.get(address_to_request, timeout=15) as resp:
                # async with retry_client.get(address_to_request, retry_attempts= 10, retry_for_statuses=statuses,
                #                             timeout=15) as resp:
                # Wait for result
                result = await resp.text()

                request_type = "vm"

                # logger.info(result)
                if "Lambda" in result.split(" "):
                    request_type = "lambda"

                # Get duration
                end = time.time()
                duration = float((end - start) * 1000)

                # Add all duration
                DURATION_LIST.append(duration)

                if request_type == "lambda":
                    logger.info(
                        f"[Lambda] Total request {total_reqs} - "
                        f"Request {num + 1}/{num_of_tasks} "
                        f"at time {arrive_time} - Response - {result} and took {duration} - Number of "
                        f"violations - {NUMBER_OF_VIOLATION}"
                    )

                elif float(duration) >= SLO_CRITERIA:
                    NUMBER_OF_VIOLATION += 1
                    VIOLATED_DURATIONS.append(duration)
                    # Info Result
                    logger.info(
                        f"Total request {total_reqs} - "
                        f"Request {num + 1}/{num_of_tasks} "
                        f"at time {arrive_time} - Response - {result} and took {duration} - Violated SLO - Number of "
                        f"violations - {NUMBER_OF_VIOLATION}"
                    )

                else:
                    # NUMBER_OF_VIOLATION += 1
                    # VIOLATED_DURATIONS.append(duration)
                    # Info Result
                    logger.info(
                        f"Total request {total_reqs} - "
                        f"Request {num + 1}/{num_of_tasks} "
                        f"at time {arrive_time} - Response - {result} and took {duration} - Number of "
                        f"violations - {NUMBER_OF_VIOLATION}"
                    )

                assert resp.status == 200

        except asyncio.TimeoutError:
            logger.info(f"Exception -> Timeout Error")

            # Get duration for this exceptional case
            end = time.time()
            duration = float((end - start) * 1000)

            # Add all duration
            DURATION_LIST.append(duration)

            NUMBER_OF_VIOLATION += 1
            VIOLATED_DURATIONS.append(duration)

            logger.info(
                f"Total request {total_reqs} - "
                f"Request {num + 1}/{num_of_tasks} "
                f"at time {arrive_time} - Response - Error and took {duration} - Violated SLO - Number of "
                f"violations - {NUMBER_OF_VIOLATION}"
            )

        except aiohttp.ClientConnectionError as e:
            logger.info(f"Exception {e} -> ClientConnectionError -> Violated SLO")
            # logger.info("Happens when reading last line")

            # Get duration for this exceptional case
            end = time.time()
            duration = (end - start) * 1000

            # Add all duration
            DURATION_LIST.append(duration)

            NUMBER_OF_VIOLATION += 1
            VIOLATED_DURATIONS.append(duration)

            logger.info(
                f"Total request {total_reqs} - "
                f"Request {num + 1}/{num_of_tasks} "
                f"at time {arrive_time} - Response - Error and took {duration} - Violated SLO - Number of "
                f"violations - {NUMBER_OF_VIOLATION}"
            )


# For Retry ClientSession
statuses = {x for x in range(100, 600)}
statuses.remove(200)
statuses.remove(429)


async def run_workload():
    # Input Information
    input_csv = lb_conf.input_csv
    logger.info(f"Start workload with {input_csv}")

    if USE_CASE_FOR_EXPERIMENT == "COLD_START":
        logger.info("Only Multiple Requests at time 1 and finish")

    # Update running time since we start workload starting from now on
    logger.info("Updating Running time before starting workload")
    inst_info: InstanceInfoClass
    for inst_id, inst_info in WHOLE_INSTANCE_INFO.items():
        inst_info.running_time = datetime.datetime.now()
    logger.info(f"\tCurrent WHOLE_INSTANCE_INFO {len(WHOLE_INSTANCE_INFO)} : {pformat(WHOLE_INSTANCE_INFO)}")

    # TEST
    # input_csv = "wits_average_130_factor_3_minute_1.csv"

    # Test -> No Scaling Factor -> We don't need scaling factor
    # if input_csv == "stable_1_minutes.csv":
    #     logger.info("\tScaling factor is None since it's test")
    #
    # # Scaling Factor will be fetched from input csv
    # else:
    #     split_string = input_csv.split("_")
    #     for i, s in enumerate(split_string):
    #         if s == "factor":
    #             scaling_factor = float(split_string[i + 1])
    #             logger.info(f"\tScaling Factor is {int(scaling_factor)}")

    # For calculation of total requests
    total_requests = 0
    # total_duration = 0
    with open(input_csv) as csv_file:

        # Open Reader
        reader = csv.reader(csv_file)

        # Open Client
        conn = aiohttp.TCPConnector(limit=None)

        async with aiohttp.ClientSession(connector=conn) as session:

            # Job keeper
            async_job = []

            # Read workload every second
            for idx, each_row in enumerate(reader):

                # arrival_time = int(each_row[0])
                arrival_time = idx + 1

                num_tasks = int(each_row[0])

                # Add request to total_requests
                total_requests += num_tasks

                # Actual Work - Send Parallel Request
                for i in range(num_tasks):
                    async_job.append(
                        asyncio.create_task(
                            send_req(
                                session, i, arrival_time, num_tasks, total_requests
                            )
                        )
                    )

                # For reading csv every second
                await asyncio.sleep(1)

            # Wait for all requests to finish
            await asyncio.gather(*async_job)


def start_workload():
    """
    Run workload using aysyncio
    :return: None
    """
    asyncio.run(run_workload())

    return


def get_number_of_requests_during_1_minute(balancer_ip_address, port_num=80):
    """
    Will return number of requests during a minute
    """
    # getting through nginx server to get current request
    address_to_req = ("http://" + balancer_ip_address + ":" + str(port_num) + lb_conf.basic_status_url)

    # set timeout to 2
    try:
        res = requests.get(address_to_req, timeout=2)
        res_split = res.text.split()

    except requests.exceptions.ConnectionError as e:
        logger.info(f"Error occurred : {e}")
        logger.info("Check balancer ip or port")
        logger.info("Currently setting number of requests to 0")
        logger.info("get_number_of_requests_during_1_minute")
        return 0

    else:
        curr_val = int(res_split[res_split.index("Reading:") - 1])
        global starting_request_number
        num_of_requests = curr_val - starting_request_number - 1
        starting_request_number = curr_val
        # logger.debug(f"\tNumber of requests is {num_of_requests}")
        logger.debug(f"\tStarting request num is now {starting_request_number}")
        assert res.status_code == 200
        logger.info("\tFinished getting number of requests during_1_minute")
        return num_of_requests


def remove_workers(instances_with_elapsed_mins: List[InstanceInfoWithCPUClass], num_of_servers_to_terminate):
    """
    Only remove number of workers given in num_of_servers_to_terminate
    """

    # logger.info(f"\t\tRemoving {num_of_servers_to_terminate} workers")

    # Revealing Information
    # logger.info(f"\t\t\tWHOLE_INSTANCE_INFO ({len(WHOLE_INSTANCE_INFO)} instances) - {WHOLE_INSTANCE_INFO}")

    # Limit number of instances
    instances_to_terminate = [i.inst_id for i in instances_with_elapsed_mins]

    # For Idle Usage -> len(instances_with_elapsed_mins) > num_of_servers_to_terminate:
    # if len(instances_with_elapsed_mins) > num_of_servers_to_terminate:
    #     for index, instance_for_termination in enumerate(instances_with_elapsed_mins):
    #
    #         # if it reaches number of instances, break
    #         instances_to_terminate.append(instance_for_termination.inst_id)
    #
    #         # If exceeds number of servers to terminate
    #         if index + 1 == num_of_servers_to_terminate:
    #             logger.info(
    #                 f"\t\t Cannot terminate more instances -> only terminate ({num_of_servers_to_terminate}) servers")
    #             break

    logger.info(f"\t\tinstances_to_terminate -> {instances_to_terminate}")

    # Number of servers to remove should be less than existing servers
    if num_of_servers_to_terminate < len(WHOLE_INSTANCE_INFO):

        # Thread keepers
        lb_detachers_threads = []

        for each_inst_id in instances_to_terminate:
            # For Consistency
            instance_workers_lock.acquire()

            # Remove inst_id from WHOLE_INSTANCE_INFO
            logger.info(f"\t\tremoving {each_inst_id} from WHOLE_INSTANCE_INFO")
            info_class: InstanceInfoClass = WHOLE_INSTANCE_INFO.pop(each_inst_id)
            logger.info(f"\t\tinstance_id list from WHOLE_INSTANCE_INFO ({len(WHOLE_INSTANCE_INFO)})-"
                        f" {WHOLE_INSTANCE_INFO.keys()}")

            TERMINATED_INSTANCE_INFO[each_inst_id] = TerminatedInstanceInfoClass(info_class.vm_class_info,
                                                                                 info_class.running_time,
                                                                                 info_class.previously_launched_instances,
                                                                                 info_class.number_of_elapsed_minutes,
                                                                                 info_class.cpu_util,
                                                                                 termination_time=datetime.datetime.now())

            # info_class: InstanceInfoClass = WHOLE_INSTANCE_INFO.pop(each_inst_id)
            # logger.info(f"\t\tinstance_id list from WHOLE_INSTANCE_INFO ({len(WHOLE_INSTANCE_INFO)})-"
            #             f" {WHOLE_INSTANCE_INFO.keys()}")
            logger.info(f"\t\tTERMINATED_INSTANCE_INFO ({len(TERMINATED_INSTANCE_INFO)})-"
                        f" {TERMINATED_INSTANCE_INFO}")

            # For Consistency
            instance_workers_lock.release()

            # Terminate and Make servers unavailable in LoadBalancer
            lb_detacher = LBDetacher(info_class.vm_class_info)
            lb_detachers_threads.append(lb_detacher)

        # Parallel execution
        logger.info("\tStart Termination")
        for thread in lb_detachers_threads:
            thread.start()

        # Wait for threads to finish
        for thread in lb_detachers_threads:
            thread.join()
        logger.info("\tFinished Termination")

        logger.info(f"\tCurrent Instances Info ({len(WHOLE_INSTANCE_INFO)}) instances:"
                    f" {pformat(WHOLE_INSTANCE_INFO)}")

    elif num_of_servers_to_terminate >= len(WHOLE_INSTANCE_INFO):

        logger.info(f"\t\tWorkers number cannot be less than resources that are to be removed")
        logger.info(f"\t\tLeaving only one worker")

        # Thread keepers
        lb_detachers_threads = []

        for each_inst_id in list(WHOLE_INSTANCE_INFO.keys())[1:]:
            # For Consistency
            instance_workers_lock.acquire()

            # Remove inst_id from WHOLE_INSTANCE_INFO
            logger.info(f"\t\t removing {each_inst_id} from WHOLE_INSTANCE_INFO")
            info_class: InstanceInfoClass = WHOLE_INSTANCE_INFO.pop(each_inst_id)
            logger.info(f"\t\t instance_id list from WHOLE_INSTANCE_INFO - {WHOLE_INSTANCE_INFO.keys()}")

            # For Consistency
            instance_workers_lock.release()

            # Terminate and Make servers unavailable in LoadBalancer
            lb_detacher = LBDetacher(info_class.vm_class_info)
            lb_detachers_threads.append(lb_detacher)

        # inst_info: InstanceInfoClass
        # for inst_id, inst_info in WHOLE_INSTANCE_INFO.items():
        #
        #     # Find instance id in WHOLE_INSTANCE_INFO and remove info
        #     if inst_id in instances_to_terminate:
        #         # For Consistency
        #         instance_workers_lock.acquire()
        #
        #         # Remove inst_id from WHOLE_INSTANCE_INFO
        #         logger.info(f"\t\t removing {inst_id} from WHOLE_INSTANCE_INFO")
        #         info_class: InstanceInfoClass = WHOLE_INSTANCE_INFO.pop(inst_id)
        #         logger.info(f"\t\t instance_id list from WHOLE_INSTANCE_INFO - {WHOLE_INSTANCE_INFO.keys()}")
        #
        #         # For Consistency
        #         instance_workers_lock.release()
        #
        #         # Terminate and Make servers unavailable in LoadBalancer
        #         lb_detacher = LBDetacher(info_class.vm_class_info)
        #         lb_detachers_threads.append(lb_detacher)

        # Parallel execution
        for thread in lb_detachers_threads:
            thread.start()
        logger.info("\t\t\t\tStarted Termination")

        # Wait for threads to finish

        for thread in lb_detachers_threads:
            thread.join()
        logger.info("\t\t\t\tFinished Termination")

    return


def determine_num_of_excess_servers(num_of_requests, num_of_vm, arrival_rate, duration):
    # logger.info("\t\tStart determine_num_of_excess_servers")

    # Initial and Exception Case
    if num_of_requests == 0:
        logger.info("\t\tSkipping since number of requests is 0")
        return 0

    if USE_CASE_FOR_EXPERIMENT == "MICROBLEND":
        """
        Assume newly launched VM can serve as if it is already running from the start
        """

        # Maximum Number of Requests for a minute which is num of servers * duration
        max_reqs_per_minute = num_of_vm * arrival_rate * duration
        logger.info(f"\t\t\tmax_reqs_with_{int(num_of_vm)}_servers : {max_reqs_per_minute}")

        # Subtract max_reqs_per_minute to current number of requests
        total_excess = num_of_requests - max_reqs_per_minute
        logger.info(f"\t\t\tTotal_excess is {total_excess}")

        # Divide with arrival rate to get number of servers====
        logger.info("\t[SCALING POLICY] - MICROBLEND")
        num_of_servers_to_launch = math.ceil(total_excess / (arrival_rate * duration))

        return num_of_servers_to_launch

    # Since there is a time for launching instances we have to calculate the gap
    elif USE_CASE_FOR_EXPERIMENT in ["ALL_VM", "OVERPROVISION"]:

        # Add partial requests for newly launched instances + Add requests for remaining numbers
        max_reqs_per_minute = get_maximum_requests_for_vms(arrival_rate, duration, num_of_vm)
        logger.info(f"\t\t\tmax_reqs_with_{int(num_of_vm)}_servers : {max_reqs_per_minute}")

        # Get total excess - Subtract max nums to current number of requests
        total_excess = num_of_requests - max_reqs_per_minute
        logger.info(f"\t\t\tTotal_excess is {total_excess}")

        # OVER_PROVISION -> 2x resources
        if OVER_PROVISION:

            logger.info("\t\t\t[SCALING POLICY] - OVERPROVISION")

            # Number of servers to use -> could be positive or negative number
            num_of_servers_to_make = math.ceil(total_excess / (arrival_rate * duration))

            if num_of_servers_to_make >= 0:
                num_of_servers_to_launch = 2 * num_of_servers_to_make
                return num_of_servers_to_launch

            # If removing workers or no provision necessary
            elif num_of_servers_to_make < 0:
                num_of_servers_to_launch = num_of_servers_to_make
                return num_of_servers_to_launch

        # NO OVER_PROVISION
        elif not OVER_PROVISION:
            logger.info("\t\t\t[SCALING POLICY] - NO OVERPROVISION ")
            num_of_servers_to_launch = math.ceil(total_excess / (arrival_rate * duration))

            return num_of_servers_to_launch

    logger.info("\t\tFinished determine_num_of_excess_servers")


@dataclass
class PreviouslyLaunchedInstancesClass:
    instance_id: str
    running_time: datetime.datetime
    gap_in_seconds: float


def get_maximum_requests_for_vms(arrival_rate, duration, num_of_vm):
    """
    Depending on the experiment type, add partial requests for previously launched instances
    1. Iterate and find previously launched instances
    2. Calculate Gap and get partial requests depending on the gap
    """

    logger.info("\t\t\tGet Maximum Requests including partial requests for previously launched instances")

    previously_launched_instances_list = []

    for i, (inst_id, info) in enumerate(WHOLE_INSTANCE_INFO.items()):
        if info.previously_launched_instances:

            logger.info(f"\t\t\t{i + 1}th instance - {inst_id}'s running time is {info.running_time}")

            # Gap in datetime format
            gap_between_running_time = datetime.datetime.now() - info.running_time

            # Gap in seconds
            gap_in_seconds = float(gap_between_running_time.total_seconds())

            logger.info(f"\t\t\t{i + 1}th instance - {inst_id} gap in seconds is {gap_in_seconds}")

            # if gap is negative since it takes more than 60 -> Skip
            if gap_in_seconds > 57:
                logger.info(
                    f"\t\t\t{i}th instance - {inst_id} gap in seconds is {gap_in_seconds} -> skip since gap is > 60")
                continue

            # Exception -> When Value is 1
            if gap_in_seconds is None:
                prev_inst_to_add = PreviouslyLaunchedInstancesClass(inst_id, info.running_time, 0)
                previously_launched_instances_list.append(prev_inst_to_add)

            else:
                prev_inst_to_add = PreviouslyLaunchedInstancesClass(inst_id, info.running_time, gap_in_seconds)
                previously_launched_instances_list.append(prev_inst_to_add)

            logger.info(f"\t\t\tpreviously_launched_instances_list with gap : {previously_launched_instances_list}")

        else:
            logger.info(f"\t\t\t{i + 1}th instance - Not Previous Instances")

    # Reqs for previous instances -> gaps * arrival rate, since gaps is for one minute
    req_for_previous_instances = arrival_rate * (sum([i.gap_in_seconds for i in previously_launched_instances_list]))
    logger.info(f"\t\t\treq_for_previous_instances : {req_for_previous_instances}")

    # Reqs for remaining instances
    req_except_recent_instances = ((num_of_vm - len(previously_launched_instances_list)) * arrival_rate * duration)
    logger.info(f"\t\t\treq_except_recent_instances : {req_except_recent_instances}")

    # Sum up above 2 variables
    max_reqs_per_minute = req_for_previous_instances + req_except_recent_instances
    logger.info(f"\t\t\tmax_reqs (total) : {max_reqs_per_minute}")

    return max_reqs_per_minute


# def get_gaps_for_newly_launched_instances():
#     gap_in_seconds_list = []
#
#     # for inst_id, inst_info in WHOLE_INSTANCE_INFO.items():
#     #     if inst_info.get('pre')
#     for inst_id in INITIAL_OR_LAUNCHED_1_MINUTE_AGO_INSTANCES:
#         inst_info = INSTANCE_DICT_INFO.get(inst_id)
#         running_time = inst_info.get("running_time")
#
#         # Due to provisioning taking more than 1 minute
#         if inst_info is None:
#             continue
#
#         else:
#             logger.info(f"\t\t{inst_id} running_time is {running_time}")
#             gap_between_running_time = datetime.datetime.now() - running_time
#             gap_in_seconds = float(gap_between_running_time.total_seconds())
#             logger.info(f"\t\t{inst_id} gap in seconds is {gap_in_seconds}")
#
#             # Value is 1 -> becomes None
#             if gap_in_seconds is None:
#                 gap_in_seconds_list.append(0)
#             else:
#                 gap_in_seconds_list.append(int(gap_in_seconds))
#
#     return gap_in_seconds_list

def make_autoscaling_decision_and_provision():
    logger.info("Make Autoscaling Decision")

    # Increment the number of elapsed time by one
    logger.info(f"\tIncrement number of elapsed time by 1")
    increment_all_instances_number_of_elapsed_mins_by_one()

    # Get statics for insanity check
    get_statistics(show_all_durations=False)

    # Get number of arrival rate for duration
    logger.info(f"\tGet number of arrival rate for duration")
    number_of_requests = get_number_of_requests_during_1_minute(balancer_ip_address=lb_ip)
    logger.info(f"\tNumber of Requests from LB is {number_of_requests}")

    # Get how many server we need to provision
    logger.info("\tDetermine number of excess servers")
    num_of_excess_server = determine_num_of_excess_servers(
        num_of_requests=number_of_requests,
        num_of_vm=len(WHOLE_INSTANCE_INFO),
        arrival_rate=lb_conf.max_requests_per_sec_per_vm,
        duration=lb_conf.duration,
    )
    logger.info(f"\t\t\tNum of excess server is {num_of_excess_server}")

    if REQUEST_TYPE == "lambda":
        logger.info("\t\tLambda -> No Provisioning needed")
        return

    # Unless using ALL LAMBDA -> Provision Resources
    else:

        # Need to scale up
        if num_of_excess_server > 0:
            scale_up_resources(num_of_excess_server)

        # No instances to remove -> terminate idle instances
        elif num_of_excess_server == 0:
            terminate_idle_resources()

        # Terminate instances with lowest cpu util
        elif num_of_excess_server < 0:

            scale_down_resources_based_on_lowest_cpu_util(num_of_excess_server)

    logger.info(f"\t\tEnd of Scaling Policy (But function inside scaling policy could be running).")

    return


def scale_down_resources_based_on_lowest_cpu_util(num_of_excess_server):
    # Number of servers to scale in  -> number of excess servers
    num_of_servers_to_scale_in = abs(num_of_excess_server)
    logger.info(f"\tRemove {num_of_servers_to_scale_in} server")

    # Set previous instance to False because at this time we didn't provision any instances
    logger.info(f"\tDoes not provision -> Setting previous instance's to False")
    for _, inst_info in WHOLE_INSTANCE_INFO.items():
        inst_info.previously_launched_instances = False

    # Insanity Check
    logger.debug(f"\tWHOLE_INSTANCE_INFO ({len(WHOLE_INSTANCE_INFO)}) is {pformat(WHOLE_INSTANCE_INFO)}")

    # Update and bring instances with sorted cpu utilization
    logger.info("\tScale in - choose the instances depending on lowest cpu util")
    instances_to_scale_in: List[InstanceInfoWithCPUClass] = get_instances_to_scale_in(WHOLE_INSTANCE_INFO, lowest=True)

    try:
        logger.info(f"\t\tInstance Candidates to Scale In ({len(instances_to_scale_in)}): {instances_to_scale_in}")
    except ValueError:
        logger.exception("Error with showing instances to scale in")

    # Choose the first {num_of_servers_to_scale_in} instances
    logger.info(f"\t\tChoose the first {num_of_servers_to_scale_in} instances")
    sliced_instances_to_scale_in = instances_to_scale_in[:num_of_servers_to_scale_in]
    logger.info(f"\t\tinstances to terminate is {sliced_instances_to_scale_in}")

    # Remove workers
    t_for_removing_resource = threading.Thread(target=remove_workers,
                                               args=[sliced_instances_to_scale_in, num_of_servers_to_scale_in])
    t_for_removing_resource.start()


def terminate_idle_resources():
    # Set previous instance to False because at this time we didn't provision any instances
    logger.info(f"\t\tDoes not provision -> Setting previous instance's to False")
    for _, inst_info in WHOLE_INSTANCE_INFO.items():
        inst_info.previously_launched_instances = False

    # Insanity Check
    logger.debug(f"\t\tWHOLE_INSTANCE_INFO {len(WHOLE_INSTANCE_INFO)} is {pformat(WHOLE_INSTANCE_INFO)}")

    # Update and Bring instances with cpu utilization less than 10 percent
    logger.info("\tDecide if we need to scale in depending on cpu util")
    instances_to_scale_in: List[InstanceInfoWithCPUClass] = get_instances_to_scale_in(WHOLE_INSTANCE_INFO)

    # Show Instances Candidate
    try:
        logger.info(f"\t\tInstances to Scale In : {instances_to_scale_in}")
    except ValueError:
        logger.exception("Exception Happened")

    # If there are instances to scale-in
    if instances_to_scale_in:

        logger.info("\tThere exist idle instances -> Scale in")

        #  candidate instances to scale in < number of excess servers  -> number of excess servers
        # if len(instances_to_scale_in) < abs(num_of_excess_server):

        # Use instances to scale in
        num_of_servers_to_scale_in = len(instances_to_scale_in)
        logger.info(f"\tRemove {num_of_servers_to_scale_in} servers")

        # Remove workers
        t_for_removing_resource = threading.Thread(target=remove_workers,
                                                   args=[instances_to_scale_in, num_of_servers_to_scale_in])
        t_for_removing_resource.start()

        #  candidate instances to scale in > number of excess servers  -> candidate instances to scale in
        # elif len(instances_to_scale_in) >= abs(num_of_excess_server):

        # Use instances to scale in
        # num_of_servers_to_scale_in = abs(num_of_excess_server)
        # sliced_instances_to_scale_in = instances_to_scale_in[:num_of_servers_to_scale_in]
        # logger.info(f"\tRemove {num_of_servers_to_scale_in} server")

        # Remove workers
        # t_for_removing_resource = threading.Thread(target=remove_workers,
        #                                            args=[instances_to_scale_in, num_of_servers_to_scale_in])
        # t_for_removing_resource.start()

    else:

        logger.info("\t\tNo idle resources")


def scale_up_resources(num_of_excess_server):
    logger.info(f"\t\t\tSet previous instances to False because we are launching new one")
    for _, inst_info in WHOLE_INSTANCE_INFO.items():
        inst_info.previously_launched_instances = False

    # Case 1 ALL VM
    if PROVISIONING_METHOD == "ONLY_VM":
        #  Bring new resources and add instances to WHOLE_INSTANCE_INFO
        t_for_new_resource = threading.Thread(target=bring_new_resource, args=[num_of_excess_server, False])
        t_for_new_resource.start()
        t_for_new_resource.join()

        logger.debug(f"\tCurrent Instances Info ({len(WHOLE_INSTANCE_INFO)}) instances:"
                     f" {pformat(WHOLE_INSTANCE_INFO)}")

    # Case 2 Use Lambda when provisioning
    if PROVISIONING_METHOD == "Hybrid":
        t_for_new_resource = threading.Thread(target=bring_new_resource, args=[num_of_excess_server])

        # Use lambda after launching instances
        t_for_new_resource.start()
        change_service_type_to_lambda()

        # After provision, change back to using VM
        t_for_new_resource.join()
        change_service_type_to_vm()

        # Insanity Check
        logger.debug(f"\t\tCurrent Instances Info ({len(WHOLE_INSTANCE_INFO)}) instances:"
                     f" {pformat(WHOLE_INSTANCE_INFO)}")
    """
                # Not at this moment
                # Case 3 Provisioning with compiler making hybrid code and Lambda
                # if should_off_load and ALREADY_OFFLOAD is False:
                #     # logging.debug(ALREADY_OFFLOAD)
                #     thread_for_offloading = threading.Thread(
                #         target=compiler.process_while_deployment, args=(whole_info,),
                #     )
                #
                #     t_for_new_resource.start()
                #     thread_for_offloading.start()
                #     thread_for_offloading.join()
                #     run_on_lambda()
                #     t_for_new_resource.join()
                #     run_on_vm()
                #     ALREADY_OFFLOAD = True
                """


def increment_all_instances_number_of_elapsed_mins_by_one():
    inst_info: InstanceInfoClass
    for inst_id, inst_info in WHOLE_INSTANCE_INFO.items():
        inst_info.number_of_elapsed_minutes += 1
    # logger.info(f"\tCurrent Instance Info ({len(WHOLE_INSTANCE_INFO)}) instances: {pformat(WHOLE_INSTANCE_INFO)}")


def start_policy():
    """
    Do autoscaling decision every duration
    """

    start_time = time.time()
    logger.info("Start Scaling Policy")

    # Skip if use_case is for test
    use_case_of_test = ["ALL_LAMBDA", "TEST_FOR_1_MINUTE_VM", "COLD_START"]
    if USE_CASE_FOR_EXPERIMENT in use_case_of_test:
        logger.info("\t ALL_LAMBDA, TEST -> Skipping Scaling Policy")
        return

    else:

        # rt = RepeatedTimer(60, make_autoscaling_decision_and_provision) # it auto-starts, no need of rt.start()
        # try:
        #     time.sleep(DURATION_IN_MIN * 60)
        # autoscaling_task = threading.Thread(target=make_autoscaling_decision_and_provision)
        # autoscaling_task.start()
        # sleep(5) # your long-running job goes here...
        # finally:
        #     rt.stop()

        while True:

            # Wake up after configured duration ====
            # time.sleep(lb_conf.duration)
            time.sleep(lb_conf.duration - ((time.time() - start_time) % lb_conf.duration))

            """
            # TODO: (For Test)
            # time.sleep(2)
            """

            # === INDEX_FOR_STOPPING_SCALING_POLICY is True after workload -> finish policy
            if INDEX_FOR_STOPPING_SCALING_POLICY:
                break

            # Make autoscaling decision ====
            autoscaling_task = threading.Thread(target=make_autoscaling_decision_and_provision)

            # Run in background ====
            autoscaling_task.start()

            """
            # TODO: (For Test)
            # autoscaling_task.join()
            # break
            """

    return


# noinspection PyUnusedLocal
def arrival_rate_decision(metrics_from_lb, arrival_rate) -> bool:
    # TODO : Later
    pass
    # # Save it to metrics_from_lb
    # metrics_from_lb.arrival_rate_operand = arrival_rate
    #
    # comparison_ops = {
    #     "<": operator.lt,
    #     "<=": operator.le,
    #     ">": operator.gt,
    #     ">=": operator.ge,
    # }
    #
    # if whole_info.offloading_whole_application:
    #     logger.info("Compare with whole application metrics")
    #     annotation_metrics = whole_info.metrics_for_whole_application
    #
    #     # If metric of arrival rate exists
    #     if metrics_from_lb.__getattribute__("arrival_rate"):
    #         arrival_rate_metric_from_lb = metrics_from_lb.__getattribute__(
    #             "arrival_rate"
    #         )
    #
    #         arrival_rate_metric_from_user = annotation_metrics.arrival_rate_operand
    #         arrival_rate_operator_user = annotation_metrics.arrival_rate_operator
    #
    #         if comparison_ops[arrival_rate_operator_user](
    #                 arrival_rate_metric_from_lb, arrival_rate_metric_from_user
    #         ):
    #             return True
    #
    #     return False
    #
    # # logger.info(
    # #     f"User Annotation from compiler Information : \n"
    # #     f"{pformat(dict(whole_info.services_for_function))}"
    # # )
    #
    # # Fetch function with rules
    # services_for_function_for_loadbalancer = whole_info.parsed_function_info_for_faas
    # # logger.info(
    # #     f"Scaling policy with metrics_from_lb : \n"
    # #     f"{pformat(dict(whole_info.services_for_function_for_loadbalancer))}"
    # # )
    #
    # service_and_metrics: FunctionWithServiceCandidate
    # for (
    #         function_name,
    #         service_and_metrics,
    # ) in services_for_function_for_loadbalancer.items():
    #
    #     service_for_function = service_and_metrics.service_candidate
    #     # Consider service contains Lambda for now
    #
    #     if service_for_function in ["Lambda", "Both"]:
    #
    #         # e.g., {'arrival_rate': [5, '>=']})
    #         rules_with_function: dict = service_and_metrics.rules_for_scaling_policy
    #
    #         if not rules_with_function:  # Dict is empty
    #             continue
    #
    #         for (
    #                 metric_name,
    #                 metric_and_operator_from_user,
    #         ) in rules_with_function.items():
    #
    #             # Metric from loadbalancer has contents
    #             if metrics_from_lb.__getattribute__(metric_name):
    #
    #                 metric_from_lb = metrics_from_lb.__getattribute__(metric_name)
    #
    #                 # Fetch function metric with comparison operator
    #                 func_metric_from_user = metric_and_operator_from_user[0]
    #                 func_operator_from_user = metric_and_operator_from_user[1]
    #
    #                 # Do comparison with metric from loadbalancer and function
    #
    #                 if comparison_ops[func_operator_from_user](
    #                         metric_from_lb, func_metric_from_user
    #                 ):  # True -> it is lambda
    #                     continue
    #
    #                 else:  # If False -> remove Lambda
    #                     whole_info.services_for_function[service_for_function].remove(
    #                         function_name
    #                     )
    #                     whole_info.services_for_function["VM"].append(function_name)
    #
    # # logger.info(
    # #     f"Service with Functions after decision : \n"
    # #     f"{pformat(dict(whole_info.services_for_function))}"
    # # )
    #
    # # sys.exit(getframeinfo(currentframe()))
    # return True


def copy_result_log():
    logger.info(f"Copying result")

    # Copy all the logging
    log_folder = module_conf.result_logfile_folder
    result_file_format = "coco_result"

    # Copy result in this directory
    log_list = glob.glob(f"{log_folder}/{result_file_format}_*.log")
    src_file = os.path.join(log_folder, "microblend_result.log")
    try:
        if RESULT_LOG_NAME:
            log_to_copy = (
                f"{log_folder}/"
                f"{result_file_format}_{len(log_list) + 2}_{RESULT_LOG_NAME}.log"
            )
            logger.info(f"\tWriting to {log_to_copy}")
            copyfile(src_file, log_to_copy)

    except NameError as e:
        logger.info(f"Exception {e} happened")

        log_to_copy = f"{log_folder}/" f"{result_file_format}_{len(log_list) + 2}.log"

        logger.info(f"\tWriting to {log_to_copy}")
        copyfile(src_file, log_to_copy)


def copy_server_log():
    """
       Copy worker (servers) information
    """
    logger.info("Copy worker (servers) information")
    workers_folder = module_conf.worker_log_folder
    worker_file_format = "workers"

    workers_info_list = glob.glob(f"{workers_folder}/{worker_file_format}_*.json")

    src_file = os.path.join(workers_folder, "workers.json")

    new_file_to_write = (
        f"{workers_folder}/{worker_file_format}_"f"{len(workers_info_list) + 1}.json"
    )

    logger.info(f"\tWriting to {new_file_to_write}")

    copyfile(src_file, new_file_to_write)


def change_service_type_to_lambda():
    logger.info("Run on Lambda")
    global REQUEST_TYPE
    REQUEST_TYPE = "lambda"


def change_service_type_to_vm():
    logger.info("Run on VM")
    global REQUEST_TYPE
    REQUEST_TYPE = "vm"


def get_instances_to_scale_in(whole_inst_info: dict, lowest=False) -> List[InstanceInfoWithCPUClass]:
    """
    get cpu utilization for 3 minutes for all instances
    Arguments - WHOLE_INSTANCE_INFO
    @rtype: list
    """

    global WHOLE_INSTANCE_INFO

    # Bring all instances with sorted lowest cpu util
    if lowest:
        logger.info(f"\t\tSort instances from lowest cpu to highest cpu")

        instances_to_scale_in = []

        client = boto3.client("cloudwatch", **CREDENTIALS, region_name="us-east-1")

        # Iterate over all instances
        inst_info: InstanceInfoClass
        for i, (instance_id, inst_info) in enumerate(whole_inst_info.items()):

            # Get number of elapsed minutes
            number_of_elapsed_minutes = inst_info.number_of_elapsed_minutes

            # Do calculation when elapsed minutes is greater 3
            if number_of_elapsed_minutes >= 3:

                try:
                    response = client.get_metric_statistics(
                        Namespace="AWS/EC2",
                        MetricName="CPUUtilization",
                        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                        StartTime=datetime.datetime.utcnow() - datetime.timedelta(seconds=180),
                        EndTime=datetime.datetime.utcnow(),
                        Period=30,
                        Statistics=["Average"],
                    )
                    # logger.debug(f"\t\t response: {response}")
                except ValueError:
                    logger.exception(f"\t\t\t\t\texception happened")

                else:
                    # List of cpu utilization every 1 minute
                    cpu_list_per_min = [x.get("Average") for x in response["Datapoints"]]
                    logger.info(f"\t\t\t\t\tcpu_list_per_min of {instance_id} is {cpu_list_per_min}")

                    try:
                        # Get average of cpu utilization
                        average_cpu_per_instance = sum(cpu_list_per_min) / len(cpu_list_per_min)

                    except ZeroDivisionError as e:
                        logger.info(f"\t\t\t\t\tException {e} -> setting cpu util to 0 ")
                        average_cpu_per_instance = 0

                    # Update this WHOLE_INSTANCE_INFO
                    inst_info.cpu_util = average_cpu_per_instance

                    # Make InstanceInfoWithCPUClass and add it to list
                    instance_with_cpu = InstanceInfoWithCPUClass(instance_id, average_cpu_per_instance)
                    logger.info(f"\t\t\t\t\tAppending {i + 1}th Instance {instance_id} to list")
                    instances_to_scale_in.append(instance_with_cpu)

            elif number_of_elapsed_minutes < 3:
                logger.info(f"\t\t\t\t\t{i + 1}th Instance {instance_id}'s elapsed time is less than 3 ")

        # Sort List depending on the cpu util
        instances_to_scale_in = sorted(instances_to_scale_in, key=lambda x: x.cpu_util)
        return instances_to_scale_in

    #  Fetch instances with cpu util less than 10 percent of total cpu
    elif not lowest:

        logger.info(f"\tFind instances to scale in when idle - Case when num of excess servers is 0")

        instances_to_scale_in = []

        client = boto3.client("cloudwatch", **CREDENTIALS, region_name="us-east-1")

        # Iterate over all instances
        inst_info: InstanceInfoClass
        for i, (instance_id, inst_info) in enumerate(whole_inst_info.items()):

            # Get number of elapsed minutes
            number_of_elapsed_minutes = inst_info.number_of_elapsed_minutes

            # Do calculation when elapsed minutes is greater 3
            if number_of_elapsed_minutes >= 3:

                logger.debug(f"\t\t{i + 1}th {instance_id} -> {inst_info.number_of_elapsed_minutes} mins elapsed")
                try:
                    response = client.get_metric_statistics(
                        Namespace="AWS/EC2",
                        MetricName="CPUUtilization",
                        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                        StartTime=datetime.datetime.utcnow() - datetime.timedelta(seconds=180),
                        EndTime=datetime.datetime.utcnow(),
                        Period=30,
                        Statistics=["Average"],
                    )
                except ValueError:
                    logger.exception(f"\t\texception happened")

                else:
                    # List of cpu utilization every 1 minute
                    cpu_list_per_min = [x.get("Average") for x in response["Datapoints"]]
                    logger.info(f"\t\tcpu_list_per_min of {i + 1}th {instance_id} is {cpu_list_per_min}")

                    try:
                        # Get average of cpu utilization
                        average_cpu_per_instance = sum(cpu_list_per_min) / len(cpu_list_per_min)

                    # When 0 -> It is idle
                    except ZeroDivisionError as e:
                        logger.info(f"\t\tException -> {e} -> Since cpu_util is 0 ")
                        average_cpu_per_instance = 0

                    # Update this WHOLE_INSTANCE_INFO
                    inst_info.cpu_util = average_cpu_per_instance

                    # If exists instance less than 10 cpu util -> add to list
                    if float(average_cpu_per_instance) <= 10:
                        logger.info(f"\t\t{i + 1}th Instance {instance_id}'s cpu util is less than 10")
                        instance_with_cpu = InstanceInfoWithCPUClass(instance_id, average_cpu_per_instance)
                        logger.info(f"\t\tAppending {i + 1}th Instance {instance_id} to list")
                        instances_to_scale_in.append(instance_with_cpu)
                    else:
                        logger.info(f"\t\t{i + 1}th Instance {instance_id}'s cpu util is greater than 10")

            elif number_of_elapsed_minutes < 3:
                logger.info(f"\t\t{i + 1}th Instance {instance_id}'s elapsed time is less than 3 ")

        return instances_to_scale_in


def fetch_instance_info():
    """
    Save Instance Id
    """

    global WHOLE_INSTANCE_INFO

    with open(module_conf.pickle_file, "rb") as file_name:
        whole_dict_info: dict = pickle.load(file_name)
    WHOLE_INSTANCE_INFO = whole_dict_info

    instances_list = []
    worker_list = []

    info: InstanceInfoClass
    for instance_id, info in WHOLE_INSTANCE_INFO.items():
        instances_list.append(instance_id)
        worker_list.append(info.vm_class_info.worker_id)

    # each_worker: VM
    # for each_worker in INSTANCE_WORKERS:
    #     # Save instance id
    #     inst_id = each_worker.instance_id
    #     instances_list.append(inst_id)
    #
    #     # Get worker list
    #     initial_work_id_list.append(each_worker.worker_id)
    #
    #     # Save it to INSTANCE_DICT_INFO
    #     INSTANCE_DICT_INFO[inst_id]["elapsed_more_than_3_mins"] = 1
    #     INSTANCE_DICT_INFO[inst_id]["worker_id"] = each_worker.worker_id
    logger.info("Fetching instances info")
    logger.info(f"\tinstances_list: {instances_list}")
    logger.info(f"\tworker_list: {worker_list}")

    return instances_list, worker_list


def save_instance_info_to_pickle(instance_info_class: dict):
    logger.info("Saving Instance Info to Pickle")
    with open("../Log/Pickle/vm_instance.pkl", "wb") as output:  # FOR TEST
        pickle.dump(instance_info_class, output, pickle.HIGHEST_PROTOCOL)
    logger.info("\tFinish Saving Instance Info to Pickle")


def get_log_name():
    """

    :return: return result log name
    """
    logger.info("Set Result Log Name")

    log_result = input("Name log name:\n")

    result_log_name = None
    if log_result == "":
        logger.info("\tNo configuration for log name")
    else:
        result_log_name = log_result

    return result_log_name


def invoke_lambda_for_cold_start():
    logger.info("Invoking lambda for preventing cold start")
    address_to_request = (
            "http://"
            + lb_conf.balancer_ip_addr
            + WORKLOAD_CHOICE.get("mat_mul")
            + "lambda"
    )
    with requests.Session() as s:
        try:
            lambda_res = s.get(address_to_request, timeout=30)
        except (
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
        ) as e:
            logger.info(f"Exception :{e}")
        else:
            # result = lambda_res.text
            # logger.info(f'{result}')
            assert lambda_res.status_code == 200
            logger.info("\tInvoked Lambda")


def get_statistics(show_all_durations=False):
    if not show_all_durations:
        logger.info("\tCheck metrics for every 1 minute")

    if show_all_durations:
        logger.debug(f"\t\tVIOLATED_DURATIONS : {VIOLATED_DURATIONS}")
        logger.debug(f"\t\tDURATION_LIST : {DURATION_LIST}")

    # logger.debug(f"\t\tWHOLE_INSTANCE_INFO : {pformat(WHOLE_INSTANCE_INFO)}")
    logger.info(f"\tCurrent Instance Info ({len(WHOLE_INSTANCE_INFO)}) instances: {pformat(WHOLE_INSTANCE_INFO)}")
    # logger.debug(f"\t\tTERMINATED_INSTANCE_INFO : {pformat(TERMINATED_INSTANCE_INFO)}")
    logger.info(
        f"\tTerminated Instance Info ({len(TERMINATED_INSTANCE_INFO)}) instances: {pformat(TERMINATED_INSTANCE_INFO)}")

    logger.debug(f"\t\tNUMBER_OF_VIOLATION : {NUMBER_OF_VIOLATION}")
    logger.debug(f"\t\tNumber of DURATION_LIST : {len(DURATION_LIST)}")
    logger.debug(f"\t\tTotal Duration : {TOTAL_DURATION}")

    try:
        logger.debug(f"\t\tSLO : {(1 - (NUMBER_OF_VIOLATION / len(DURATION_LIST)))}")
        logger.debug(f"\t\tAverage Request Per Second: {len(DURATION_LIST) / TOTAL_DURATION}")
        logger.debug(f"\t\tMedian of DURATION_LIST : {np.median(DURATION_LIST)}")
        logger.debug(f"\t\tMean of DURATION_LIST : {np.mean(DURATION_LIST)}")
    except ZeroDivisionError:
        logger.debug("\t\tZero Exception")


"""
Start of the code
"""

WHOLE_INSTANCE_INFO = defaultdict(InstanceInfoClass)
TERMINATED_INSTANCE_INFO = defaultdict(InstanceInfoClass)

# For keeping initial instances
INITIAL_OR_LAUNCHED_1_MINUTE_AGO_INSTANCES = []

# Declare and Bring Logger
logger = return_logger()

# Empty Result Log for filling up a new one
empty_result_log()

# Get Initial number of requests
starting_request_number = get_initial_request()

# print(result)


# sys.exit(getframeinfo(currentframe()))  # TODO: Stop at this if you want only see if lb is running

if __name__ == "__main__":
    # Use case and result log name
    logger.info(f"USE_CASE_FOR_EXPERIMENTS : {USE_CASE_FOR_EXPERIMENT}")
    logger.info(f"RESULT_LOG_NAME is {RESULT_LOG_NAME}")

    # Experiment Order Phase 1 -> Phase 2,3

    # TODO: Phase 1

    # === 1. Do some user annotation and putting pragma

    # f_name = module_conf.f_name
    # bench_dir = module_conf.bench_dir
    # # """
    # # === Use compiler later
    # # whole_info = compiler_simplified.process_before_deploy(f_name, bench_dir)
    # # """
    # # === 2. Get initial experiment setup, make new resources
    # # 2.1 Make previous workers unavailable
    # make_all_workers_unavailable()
    # # === 2.2 Terminate previous workers
    terminate_previous_instances()
    # # === 2.3 Remove worker information from worker log
    # remove_contents_in_worker_log()
    # # === 2.4 Start new initial resources
    # bring_new_resource(lb_conf.init_number_of_instance, init_phase=True)
    # # === 3. Save instance info to pickle format so later we can experiment with initial previously launched instances
    # save_instance_info_to_pickle(WHOLE_INSTANCE_INFO)
    # # === 4. Stop right here and comment Step 3, 4 and 5 -> move on to 6
    sys.exit(getframeinfo(currentframe()))  # TODO : Make sure finish here

    # TODO: Phase 2

    # 5. ===== Fetch instance information =====
    instance_id_list, worker_id_list = fetch_instance_info()
    logger.info(f"\tINITIAL INSTANCE_WORKERS : {instance_id_list}")
    logger.info(f"\tWHOLE_INSTANCE_INFO ({len(WHOLE_INSTANCE_INFO)}) instances: {pformat(WHOLE_INSTANCE_INFO)}")
    # 6.1 ===== Remove unnecessary previous instances ====
    terminate_previous_instances(instance_id_list)
    # 6.2 ===== make only initial resources available ====
    make_all_workers_unavailable(worker_id_list)
    logger.info(f"\tWHOLE_INSTANCE_INFO ({len(WHOLE_INSTANCE_INFO)}) instances: {pformat(WHOLE_INSTANCE_INFO)}")

    # sys.exit(getframeinfo(currentframe())) # For Test

    # TODO: Phase 3 -> Workload and Policy

    # 7.1 Workload and Policy concurrently =====
    workload_task = threading.Thread(target=start_workload)

    # 7.2 Policy =====
    policy_task = threading.Thread(target=start_policy)

    # 7.3 Start workload =====
    workload_task.start()

    # 7.4 Start policy =====
    policy_task.start()

    # 7.5 Wait for workload to finish =====
    workload_task.join()
    logger.info("Ended Workload\n")

    # 7.6 Stop policy after finishing workload =====
    INDEX_FOR_STOPPING_SCALING_POLICY = True
    policy_task.join()
    logger.info("Ended Policy")

    # logger.info(f"INSTANCE_DICT_INFO {INSTANCE_DICT_INFO}")
    get_statistics(show_all_durations=True)

    # 8 Copy result and server log =====
    copy_result_log(), copy_server_log(),  # sys.exit(getframeinfo(currentframe()))

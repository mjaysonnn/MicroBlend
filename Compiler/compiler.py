import ast
import copy
import itertools
import logging
import os
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pprint import pformat
from typing import List, Any, Dict, Union

import astor
import boto3


class ModuleConfig:
    """
    Implements a class for storing the configuration of the compiler module.
    """
    benchmark_dir = "../BenchmarkApplication"
    file_name = ""


module_config = ModuleConfig()
benchmark_dir = module_config.benchmark_dir
file_name = module_config.file_name

# Logging Configuration
modules_for_removing_debug = ["urllib3", "s3transfer", "boto3", "botocore", "urllib3", "requests", "paramiko"]

for name in modules_for_removing_debug:
    logging.getLogger(name).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.propagate = False
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-6s] [%(filename)s:%(lineno)-4s]  %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

CREDENTIALS = {}
s3_client = boto3.client("s3", **CREDENTIALS)
s3 = boto3.resource("s3", **CREDENTIALS)


@dataclass
class TargetGoalConfig:
    cost: float = 2.3
    performance: float = 1.2


def return_vm_types():
    """
    return list of vm types
    :return:
    """
    return [
        "c5.large",
        "c3.large",
        "m4.large",
        "m3.large",
        "c5.2xlarge",
        "c3.2xlarge",
    ]


@dataclass
class VMRuntimeConfig:
    vm_instance_selection_list: List = field(default_factory=return_vm_types)
    region_name: str = "us-east-1"


@dataclass
class FunctionWithServiceCandidate:
    service_candidate: str = None
    merge_function: str = None
    rules_for_scaling_policy: dict = field(default_factory=defaultdict)


@dataclass
class UserInputClass:
    original_file: str = "media_service.py"
    target_goal_config: TargetGoalConfig = field(default_factory=TargetGoalConfig)
    vm_runtime_config: VMRuntimeConfig = field(default_factory=VMRuntimeConfig)


def terminate_instance(instance_id, region_name="us-east-1"):
    ec2_client = boto3.client("ec2", region_name=region_name, **CREDENTIALS)
    ec2_client.terminate_instances(InstanceIds=[instance_id, ], )
    logger.info(f"Terminating instance - {instance_id}")


def user_annotation_for_application() -> UserInputClass:
    """
    Make info from user annotation
    1. Original File
    2. VM Runtime configuration
    3. Target Goal
    3. Function service with parameters
    """

    # Fist make dataclass and save default values
    user_input_class = UserInputClass()

    # Input source code
    source_code = input("enter your application code:\n")
    if source_code != "":
        user_input_class.original_file = source_code

    # Target Goal - Cost or Performance
    target_goal_config = TargetGoalConfig()
    cost_goal = input("Enter the cost goal ($):\n")
    if cost_goal != "":
        target_goal_config.cost = int(cost_goal)

    # Performance goal
    performance_goal = input("Enter the performance goal (ms):\n")
    if performance_goal != "":
        target_goal_config.performance = float(performance_goal)

    # Add target_goal_config to user_input_class
    user_input_class.target_goal_config = target_goal_config

    # VM Runtime Configuration, VM AWS RegionName
    vm_runtime_config = VMRuntimeConfig()
    vm_instance_types = input(f"VM Selection: select VM types : {return_vm_types()}\n")
    if vm_instance_types != "":
        vm_selection_list = [
            x.strip() for x in vm_instance_types.split(",") if vm_instance_types
        ]
        vm_runtime_config.vm_instance_selection_list = vm_selection_list

    # AWS Region
    vm_region_name = input("Choose aws region for VM:\n")
    if vm_region_name != "":
        vm_runtime_config.region_name = vm_region_name

    # Add vm_runtime_config to user input class
    user_input_class.vm_runtime_config = vm_runtime_config
    logging.debug(f"user_input_class is \n {pformat(user_input_class.__dict__)}")

    return user_input_class


def generate_original_and_copied_ast_info(original_code_script, original_filename):
    """
    save original code and module object and copied module object
    """
    # Make AST object
    original_ast: ast.Module = ast.parse(original_code_script.read())

    # Create a deep copy of the AST for analysis
    module_for_analysis = copy.deepcopy(original_ast)

    # Return dataclass
    return WholeASTInfoClass(file_name=original_filename, original_code_module=original_ast,
                             copied_module_for_analysis=module_for_analysis)


@dataclass
class ASTObjectInfoClass:
    line_number: int
    ast_object: Any


@dataclass(order=True)
class ImportInfoClass(ASTObjectInfoClass):
    import_name: str = None

    from_python_script: bool = False
    from_import: list = field(default_factory=list)

    import_as: list = field(default_factory=list)
    assign_targets_from_non_func: set = field(default_factory=set)


def give_services_input():
    return "{IaaS, FaaS}"


@dataclass
class ServiceTypeForFunction:
    service_type: List = field(default_factory=give_services_input)


@dataclass
class LambdaConfig:
    memory: int = 512
    timeout: int = 100
    runtime_version: str = "py37"
    func_name: str = None


@dataclass
class FunctionMetricRules:
    cpu_util_use_default_value: bool = True
    cpu_util_operand: float = 40
    cpu_util_operator: str = ">"

    memory_util_use_default_value: bool = True
    memory_util_operand: float = 30
    memory_util_operator: str = ">"

    arrival_rate_use_default_value: bool = True
    arrival_rate_operand: int = 5
    arrival_rate_operator: str = ">"


@dataclass
class ModuleImportClass:
    ast_object: Any


@dataclass(order=True)
class NonFunctionInfoClass(ASTObjectInfoClass):
    object_type: str = None
    imports: set = field(default_factory=set)
    assignments: list = field(default_factory=list)
    description: str = None


@dataclass(order=True)
class FunctionDefinition(ASTObjectInfoClass):
    """
    Function definition object
    """
    # Basic Information
    func_name: str = False
    initial_pragma: str = None
    lambda_group_name: str = "Default"

    # Import and Reference Information
    imports: set = field(default_factory=set)
    assigned_targets: list = field(default_factory=list)

    # Function Details
    parameters: list = field(default_factory=list)
    return_values: list = field(default_factory=list)
    return_values_ast: Any = None

    # Lambda Configuration
    lambda_config: LambdaConfig = field(default_factory=LambdaConfig)
    function_metric_rules: FunctionMetricRules = field(default_factory=FunctionMetricRules)
    lambda_handler_call: Union[ast.Assign, ast.Expr] = field(default_factory=list)


@dataclass
class FunctionCallInfo(ASTObjectInfoClass):
    """
    Function call object
    """
    object_type: str = None
    caller_object: ast.FunctionDef = None
    caller_object_name: str = None
    callee_object: ast.Call = None
    callee_object_name: str = None
    assign_targets: list = field(default_factory=list)
    call_func_params: list = field(default_factory=list)


@dataclass
class FunctionCallInsideIfMain(ASTObjectInfoClass):
    object_type: str = None

    callee_object: ast.Call = None
    callee_object_name: str = None


@dataclass
class LambdaBasedFunctionCallInfoClass:
    copied_func_call_info: FunctionCallInfo
    original_func_call_info: FunctionCallInfo


@dataclass
class CompilerGeneratedLambda:
    lambda_name: str
    original_func_name: str
    lambda_module: ast.Module
    lambda_handler_func_object: ast.FunctionDef
    original_ast_object: ast.FunctionDef
    lambda_event_input_objs: list = field(default_factory=list)
    input_parameter: List[ast.Assign] = field(default_factory=list)
    import_info_dict: dict = field(default_factory=defaultdict)
    return_objects: list = field(default_factory=list)
    import_name_list: list = field(default_factory=list)  # for offloading whole app


@dataclass
class MergedCompilerGeneratedLambda:
    lambda_group_name: str
    lambda_name_list: list
    lambda_module: ast.Module
    lambda_handler_func_object: ast.FunctionDef
    original_ast_object_list: list
    lambda_event_input_objs: dict = field(default_factory=defaultdict)
    input_parameter: List[ast.Assign] = field(default_factory=list)
    return_objects: list = field(default_factory=list)
    import_name_list: list = field(default_factory=list)
    lambda_input_per_if_statement: dict = field(default_factory=defaultdict)
    parse_input_per_if_statement: dict = field(default_factory=defaultdict)
    return_obj_per_if_statement: dict = field(default_factory=defaultdict)


@dataclass
class CompilerGeneratedLambdaForIFMainObject:
    lambda_name: str
    lambda_based_module_ast_object: ast.Module
    original_if_main_ast_object: ast.FunctionDef


@dataclass
class LambdaHandlerModuleObject:
    module_object = ast.Module


@dataclass
class AnnotationClass:
    lambda_pragma: str = "# Pragma BEU FaaS"
    vm_pragma: str = "# Pragma BEU IaaS"
    hybrid_pragma: str = "# Pragma BEU {IaaS, FaaS}"


@dataclass
class LambdaDeployInfo:
    # lambda_name: str
    original_func_name: str
    lambda_zip_name: str
    lambda_name_to_make_on_aws: str
    lambda_arn: str
    handler_name: str
    zip_file_name_in_s3: str
    aws_role: str = "arn:aws:iam::206135129663:role/mj_boss"
    region: str = "us-east-1"
    runtime: str = "python3.7"
    memory_size: int = 512
    time_out: int = 100


@dataclass
class FunctionWithPragma:
    function_name: str = None
    pragma: str = "Both"


@dataclass
class WholeASTInfoClass:
    file_name: str

    # Module
    original_code_module: ast.Module
    copied_module_for_analysis: ast.Module
    hybrid_code_module: ast.Module = None
    annotated_code_module_for_user: ast.Module = None

    # Offloading info
    offloading_whole_application: bool = False
    module_info_for_offloading_whole_app: CompilerGeneratedLambda = None
    deployment_name_for_offloading_whole_app: str = None

    # Import info
    import_information: Dict[str, ImportInfoClass] = field(default_factory=dict)
    boto3_and_json_imports: Dict[str, ModuleImportClass] = field(default_factory=dict)

    # Function and Non-Function info
    non_function_object_info: Dict[Union[ast.Expr, ast.Assign], NonFunctionInfoClass] = field(default_factory=dict)
    function_information: Dict[str, FunctionDefinition] = field(default_factory=dict)
    function_list_for_annotation: List[str] = field(default_factory=list)
    sort_by_lambda_group: Dict[str, List] = field(default_factory=dict)
    function_call_info_class: Dict[Union[ast.Expr, ast.Assign], FunctionCallInfo] = field(default_factory=dict)
    func_call_using_lambda: list = field(default_factory=list)
    combined_func_call_using_lambda: Dict[str, List] = field(default_factory=dict)
    lambda_invoke_function: ast.FunctionDef = None

    # if __main__ info
    objects_inside_if_main: dict = field(default_factory=defaultdict)
    function_call_inside_if_main: dict = field(default_factory=defaultdict)
    if_main_object: NonFunctionInfoClass = None

    # Lambda related info
    lambda_function_info: dict = field(default_factory=defaultdict)
    lambda_deployment_zip_info: dict = field(default_factory=defaultdict)
    lambda_config_for_whole_application: LambdaConfig = None
    map_func_to_func_arn_object: ast.Assign = None
    lambda_handler_module_dict: dict = field(default_factory=defaultdict)
    parsed_function_info_for_faas: dict = field(default_factory=defaultdict)


def extract_user_annotations(whole_info_class: WholeASTInfoClass):
    """
    Let users choose which functions they want to annotate
    """

    # logger.info("Find pragma")

    # Get the module for analysis
    analyzed_module = whole_info_class.copied_module_for_analysis

    # Dictionary to store information about functions
    func_information = defaultdict(FunctionDefinition)
    functions_for_user_annotation = []

    # Inspect each node in the module for annotations
    for child_node in ast.iter_child_nodes(analyzed_module):
        if isinstance(child_node, ast.FunctionDef):
            function_def = FunctionDefinition(
                func_name=child_node.name,
                ast_object=child_node,
                line_number=child_node.lineno)
            function_def = find_pragma_in_function(child_node, function_def)
            # logger.debug(f"function_def is {function_def}")

            # Store function's information
            func_information[child_node.name] = function_def

            # If no user annotation, compiler will provide annotation
            if function_def.initial_pragma is None:
                functions_for_user_annotation.append(child_node.name)

            # Main function and Pragma BEU FaaS  -> Offloading whole app
            if function_def.initial_pragma == "FaaS" and function_def.func_name == "main":
                logger.info("\tPreparing to offload the entire application...")
                whole_info_class.offloading_whole_application = True

    # Update the dataclass attributes
    whole_info_class.function_information = dict(func_information)
    whole_info_class.function_list_for_annotation = functions_for_user_annotation


def choose_functions_to_annotate(func_names_to_annotate):
    return input(
        f"Choose functions you want to annotate "
        f"(f1, f2) or whole : {func_names_to_annotate}\n"
    )


def set_config_for_each_function(
        each_func_name, map_func_name_to_function_info_class, service_info_per_functions
):
    # Lambda, VM, Both
    func_service = provide_user_with_service_type(each_func_name)  # FIXME :test

    # Get FunctionDefinition for each function
    function_info: FunctionDefinition = map_func_name_to_function_info_class[
        each_func_name
    ]
    # Write service type for each function
    if func_service.strip():

        function_info.service_type_for_function.service_type = func_service

        save_service_per_function(
            service_info_per_functions, func_service, each_func_name
        )

        # In case of using FaaS, need to set lambda runtime conf
        if func_service in ("{IaaS, FaaS}", "FaaS"):
            # Lambda configuration
            provide_user_with_lambda_config(function_info, each_func_name)

            # Scaling Policy Metrics
            provide_user_with_cpu_utilization(function_info)
            provide_user_with_arrival_rate(function_info)
            provide_user_with_memory_utilization(function_info)


def set_configuration_for_whole_app(
        map_func_name_to_function_info_class, whole_info_class
):
    logger.info("Annotating Whole Application")

    whole_info_class.offloading_whole_application = True
    whole_info_class.function_information = dict(map_func_name_to_function_info_class)

    logger.info("Setting Metrics for whole application")

    lambda_config = set_lambda_config_for_whole_app()
    whole_info_class.lambda_config_for_whole_application = lambda_config

    metrics_for_whole_app = set_metrics_for_whole_app()
    whole_info_class.metrics_for_whole_application = metrics_for_whole_app


def find_pragma_in_function(child, func_def: FunctionDefinition):
    """
    Find pragma in function definition
    """
    for child_node in ast.iter_child_nodes(child):
        if isinstance(child_node, ast.Expr) and isinstance(child_node.value, ast.Str):
            split_comments = child_node.value.s.strip().split()
            it = iter(split_comments)
            func_metrics = FunctionMetricRules()
            iterate_comments_and_save_pragma(func_def, func_metrics, it)
    return func_def


def iterate_comments_and_save_pragma(func_def, function_metric_rules, it):
    """
    Iterate comments and save pragma
    """
    try:
        while True:
            c = next(it)
            if c == "Pragma" and next(it) == "BEU":
                func_def.initial_pragma = next(it)

            if c == "Combine":
                func_def.lambda_group_name = next(it)

            if c == "Metric":
                metric_type = next(it)

                if "arrival_rate" in metric_type:
                    function_metric_rules.arrival_rate_use_default_value = False
                    function_metric_rules.arrival_rate_operator = next(it)
                    function_metric_rules.arrival_rate_operand = next(it)[:-1]

                elif "cpu_util" in metric_type:
                    function_metric_rules.cpu_util_use_default_value = False
                    function_metric_rules.cpu_util_operator = next(it)
                    function_metric_rules.cpu_util_operand = next(it)[:-1]

                elif "memory_util" in metric_type:
                    function_metric_rules.memory_util_use_default_value = False
                    function_metric_rules.memory_util_operator = next(it)
                    function_metric_rules.memory_util_operand = next(it)[:-1]

    except StopIteration:
        pass

    func_def.function_metric_rules = function_metric_rules


def see_if_main_function_has_pragma(child, split_comments, whole_info_class):
    if child.name == "main":
        whole_info_class.pragma_exist_in_main_function = True
        whole_info_class.raw_pragma_and_metrics_in_main_function = split_comments


def set_metrics_for_whole_app():
    logger.info("Setting metrics for whole application")

    metrics_for_whole_app = FunctionMetricRules()

    # cpu_util = input('cpu_util (80):\n')
    # cpu_util_operator = input(f'operator (>=):\n')
    cpu_util, cpu_util_operator = "", ""

    if cpu_util.strip():
        metrics_for_whole_app.cpu_util_operand = int(cpu_util)
        metrics_for_whole_app.cpu_util_use_default_value = False

    if cpu_util_operator.strip():
        metrics_for_whole_app.cpu_operator = cpu_util_operator
        metrics_for_whole_app.cpu_util_use_default_value = False

    arrival_rate, arrival_rate_operator = "5", ">="

    if arrival_rate.strip():
        metrics_for_whole_app.arrival_rate_operand = int(arrival_rate)
        metrics_for_whole_app.arrival_rate_use_default_value = False

    if arrival_rate_operator.strip():
        metrics_for_whole_app.arrival_rate_operator = arrival_rate_operator
        metrics_for_whole_app.arrival_rate_use_default_value = False

    memory_util, memory_util_operators = "", ""

    if memory_util.strip():
        metrics_for_whole_app.memory_util_operand = int(memory_util)
        metrics_for_whole_app.memory_util_use_default_value = False

    if memory_util_operators.strip():
        metrics_for_whole_app.memory_util_operator = memory_util_operators
        metrics_for_whole_app.memory_util_use_default_value = False

    return metrics_for_whole_app


def set_lambda_config_for_whole_app():
    logger.info("Lambda configuration for whole application")
    lambda_config_for_whole_app = LambdaConfig()

    # lambda_memory = input(f"lambda_config : [memory_size (512):\n")
    lambda_memory = ""
    if lambda_memory.strip():
        lambda_config_for_whole_app.memory = lambda_memory

    lambda_timeout = ""
    if lambda_timeout.strip():
        lambda_config_for_whole_app.timeout = lambda_timeout

    lambda_runtime = ""
    if lambda_runtime.strip():
        lambda_config_for_whole_app.runtime_version = lambda_runtime

    lambda_name = "resnet18_lambda"  # FIXME: for test
    if lambda_name.strip():
        lambda_config_for_whole_app.func_name = lambda_name

    return lambda_config_for_whole_app


def provide_user_with_service_type(each_func_name):
    # Input for Service Type
    mixture_pragma = "IaaS, FaaS, {IaaS, FaaS}(default)"

    return input(f"{each_func_name} configuration : "
                 f"service type - {mixture_pragma}:\n "
                 )


def provide_user_with_memory_utilization(function_info):
    memory_util, memory_util_operators = "", ""  # FIXME: for test -> comment later

    if memory_util != "":
        function_info.function_metric_rules.memory_util_operand = int(memory_util)
        function_info.function_metric_rules.memory_util_use_default_value = False

    if memory_util_operators != "":
        function_info.function_metric_rules.memory_util_operator = memory_util_operators
        function_info.function_metric_rules.memory_util_use_default_value = False


def provide_user_with_arrival_rate(function_info):
    # arrival_rate = input(f'{each_func_name} target metrics : arrival_rate (5) :\n')
    # arrival_rate_operator = input(f'{each_func_name} target metrics :operator (>=):\n')
    # FIXME: for test -> comment later
    arrival_rate, arrival_rate_operator = "5", ">="

    if arrival_rate != "":
        function_info.function_metric_rules.arrival_rate_operand = int(arrival_rate)
        function_info.function_metric_rules.arrival_rate_use_default_value = False

    if arrival_rate_operator != "":
        function_info.function_metric_rules.arrival_rate_operator = (
            arrival_rate_operator
        )
        function_info.function_metric_rules.arrival_rate_use_default_value = False


def provide_user_with_cpu_utilization(function_info):
    # cpu_util = input(f'{each_func_name}:metrics : cpu_util (80):\n')
    # cpu_util_operator = input(f'{each_func_name}:metrics : operator (>=):\n')
    cpu_util, cpu_util_operator = "", ""  # FIXME: for test -> comment later

    if cpu_util != "":
        function_info.function_metric_rules.cpu_util_operand = int(cpu_util)
        function_info.function_metric_rules.cpu_util_use_default_value = False

    if cpu_util_operator != "":
        function_info.function_metric_rules.cpu_operator = cpu_util_operator
        function_info.function_metric_rules.cpu_util_use_default_value = False


def provide_user_with_lambda_config(function_info, each_func_name):
    # Memory
    # lambda_config_memory = input(f'{each_func_name} lambda_config '
    #                              f': [memory_size (512):\n')
    lambda_config_memory = ""  # FIXME: for test -> comment later
    if lambda_config_memory.strip():
        function_info.lambda_config.memory = lambda_config_memory

    # timeout
    # lambda_config_timeout = input(f'{each_func_name} lambda_config '
    #                               f': [time_out (100):\n')
    lambda_config_timeout = ""  # FIXME: for test
    if lambda_config_timeout.strip():
        function_info.lambda_config.timeout = lambda_config_timeout

    # Lambda Runtime
    # lambda_config_runtime = input(f'{each_func_name} lambda_config : '
    #                               f'[runtime (py37):\n')
    lambda_config_runtime = ""  # FIXME: for test
    if lambda_config_runtime.strip():
        function_info.lambda_config.runtime_version = lambda_config_runtime


def save_service_per_function(service_for_functions, func_service, each_func_name):
    """
    Save service type for each function
    """
    if func_service == "IaaS":
        service_for_functions["VM"].append(each_func_name)

    elif func_service == "FaaS":
        service_for_functions["Lambda"].append(each_func_name)


def parse_info_for_functions(whole_ast_info: WholeASTInfoClass):
    """
    make information : function and function rules for parsing to loadbalancer
    """

    logger.info("Parse info for functions")

    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app] : Info exist in function definition")

    else:
        parse_and_save_info_for_each_function(whole_ast_info)


def parse_and_save_info_for_each_function(whole_ast_info: WholeASTInfoClass):
    """
    Parse and save information for each function
    :param whole_ast_info: The WholeASTInfoClass containing the information for the whole AST
    """
    function_info_dict = whole_ast_info.function_information
    services_for_function = whole_ast_info.function_list_with_services

    function_with_service_candidate = {}

    for service_type, function_list in services_for_function.items():
        for each_function_name in function_list:
            function_info: FunctionDefinition = function_info_dict.get(each_function_name)
            if function_info.initial_pragma:
                function_with_service_candidate[each_function_name] = FunctionWithServiceCandidate(
                    service_candidate=function_info.initial_pragma,
                    merge_function=function_info.merge_function,
                )
            else:
                metrics_for_scaling_policy = function_info.function_metric_rules
                metrics_dict = defaultdict(list)
                if not metrics_for_scaling_policy.memory_util_use_default_value:
                    metrics_dict["memory_util"].append(metrics_for_scaling_policy.memory_util_operand)
                    metrics_dict["memory_util"].append(metrics_for_scaling_policy.memory_util_operator)
                if not metrics_for_scaling_policy.cpu_util_use_default_value:
                    metrics_dict["cpu_util"].append(metrics_for_scaling_policy.cpu_util_operand)
                    metrics_dict["cpu_util"].append(metrics_for_scaling_policy.cpu_util_operator)
                if not metrics_for_scaling_policy.arrival_rate_use_default_value:
                    metrics_dict["arrival_rate"].append(metrics_for_scaling_policy.arrival_rate_operand)
                    metrics_dict["arrival_rate"].append(metrics_for_scaling_policy.arrival_rate_operator)
                function_with_service_candidate[each_function_name] = FunctionWithServiceCandidate(
                    service_candidate=service_type,
                    rules_for_scaling_policy=metrics_dict,
                )

    logging.debug("\n" + pformat(dict(function_with_service_candidate)))
    whole_ast_info.parsed_function_info_for_faas = function_with_service_candidate


def show_result(whole_info):
    logger.debug("\n" + pformat(whole_info.__dict__))


def put_faas_pragma(beu_pragma_to_add, func_info_class):
    """
    Put FaaS pragma in case of [IaaS, FaaS] or FaaS. Also will put scaling policy metrics
    """
    metrics_for_lb = func_info_class.function_metric_rules

    # Add scaling policy metrics if user annotation
    metrics_to_add = []
    if not metrics_for_lb.arrival_rate_use_default_value:
        metrics_to_add.append(
            f"arrival_rate {metrics_for_lb.arrival_rate_operator} {metrics_for_lb.arrival_rate_operand}"
        )

    if not metrics_for_lb.cpu_util_use_default_value:
        metrics_to_add.append(
            f"cpu_util {metrics_for_lb.cpu_operator} {metrics_for_lb.cpu_util_operand}"
        )

    if not metrics_for_lb.memory_util_use_default_value:
        metrics_to_add.append(
            f"memory_util {metrics_for_lb.memory_util_operator} {metrics_for_lb.memory_util_operand}"
        )

    metrics_str = " ".join(metrics_to_add)
    beu_pragma_to_add = f"\n\t{beu_pragma_to_add} {metrics_str}\n\t"

    for child in ast.iter_child_nodes(func_info_class.ast_object):
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Str):
            child.value.s += beu_pragma_to_add + "\n\t"
            break
    else:
        pragma_node = ast.Expr(value=ast.Str(s=beu_pragma_to_add))
        func_info_class.ast_object.body.insert(0, pragma_node)

    ast.fix_missing_locations(func_info_class.ast_object)


def put_iaas_pragma(func_info_class, pragma_class):
    """
    Put IaaS Pragma -> No metrics needed
    """
    for child in ast.iter_child_nodes(func_info_class.ast_object):
        if isinstance(child, ast.Expr) and isinstance(child.value, ast.Str):
            child.value.s += pragma_class.vm_pragma + "\n\t"
            break
    else:
        pragma_node = ast.Expr(value=ast.Str(s=pragma_class.vm_pragma))
        func_info_class.ast_object.body.insert(0, pragma_node)

    ast.fix_missing_locations(func_info_class.ast_object)


def put_pragma_comment_in_func(func_info_class: FunctionDefinition):
    """
    Put Pragma comment in function
    """
    pragma_class = AnnotationClass()

    if func_info_class.service_type_for_function.service_type == "IaaS":
        put_iaas_pragma(func_info_class, pragma_class)


@dataclass
class ModuleConfigClass:
    # Basic information
    file_name: str

    # Log configurations
    result_log_path: str = "log_folder/result_log/microblend_result.log"
    worker_log_path: str = "log_folder/workers_from_lb/workers.json"

    # Compiler directories and paths
    benchmark_directory: str = "../Workload"
    module_directory: str = "import_modules"
    output_directory: str = "output"

    # Derived paths from output directory
    lambda_code_path: str = f"{output_directory}/lambda_codes"
    deployment_zip_path: str = f"{output_directory}/deployment_zip_dir"
    hybrid_code_directory: str = f"{output_directory}/hybrid_vm"

    # File names and S3 buckets
    hybrid_code_filename: str = "compiler_generated_hybrid_code"
    hybrid_code_bucket: str = "microblend-hybrid-bucket-mj"
    lambda_handler_zip_bucket: str = "faas-code-deployment-bucket"


class ImportAndFunctionAnalyzer(ast.NodeVisitor):
    """
    Analyze import and function
    """

    def __init__(self, ast_info: WholeASTInfoClass, config: ModuleConfigClass):
        self.ast_info = ast_info
        self.file_name = ast_info.file_name
        self.target_source_code = ast_info.copied_module_for_analysis
        self.config_info = config

    def _extract_import_name(self, alias):
        """
        Extract the main part of an import statement, handling "as" aliases.
        E.g., "import numpy as np" -> "numpy"
        """
        return alias.name.split(".")[0] if "." in alias.name else alias.name

    def visit_Import(self, node):
        """
        Handle import numpy or import numpy as np
        """
        primary_import_name = self._extract_import_name(node.names[0])
        import_info = ImportInfoClass(line_number=node.lineno, ast_object=node)

        for alias in node.names:
            import_name = self._extract_import_name(alias)
            import_info.import_name = import_name

            if alias.asname:
                import_info.import_as.append(alias.asname)

        self.ast_info.import_information[primary_import_name] = import_info
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Handle from multiprocessing import Pool
        Handle from import
        """
        import_info_class = ImportInfoClass(line_number=node.lineno, ast_object=node)

        node_module_name = node.module.split(".")[0]
        import_info_class.import_name = node_module_name

        for alias in node.names:
            if alias.asname:
                import_info_class.from_import.append(alias.asname)
            else:
                import_info_class.from_import.append(alias.name)

        self.ast_info.import_information[node_module_name] = import_info_class
        self.generic_visit(node)

    def _fetch_python_script_names(self, target_file_name):
        """
        Fetch all Python script names in the benchmark directory except the target_file_name.
        """

        python_script_names = []

        # Store the current working directory to return to it later
        initial_directory = os.getcwd()

        try:
            # Change the directory to the benchmark directory specified in the config
            os.chdir(self.config_info.benchmark_directory)

            # Use a list comprehension to fetch all the python script names
            # Exclude the target file and its base name
            python_script_names.extend(
                file.replace(".py", "")
                for file in os.listdir()
                if (
                        file.endswith(".py")
                        and file != os.path.basename(target_file_name)
                        and file != target_file_name
                )
            )
        finally:
            # Return to the initial directory regardless of any exceptions
            os.chdir(initial_directory)

        return python_script_names

    def find_imports_from_python_scripts(self, target_file_name):
        """
        Iterate through benchmark directory and find code that is ".py"
        """
        # Get a list of all python scripts in the benchmark directory excluding the target_file_name
        python_script_imports = self._fetch_python_script_names(target_file_name)

        logger.debug(python_script_imports)

        # For each import in the AST info, check if it matches a name from python_script_imports
        # If it does, set the 'from_python_script' flag to True for that import
        for _, import_info in self.ast_info.import_information.items():
            if import_info.import_name in python_script_imports:
                import_info.from_python_script = True

    def find_function_calls_within_assignment(
            self, assign_targets, assignment_node: ast.Assign, function_node: ast.FunctionDef
    ):
        """
        Find function call in assign object in function object
        """

        # Check if the assigned value is a function call or a binary operation
        assigned_value = assignment_node.value
        if not isinstance(assigned_value, (ast.Call, ast.BinOp)):
            return

            # Scan all child nodes of the assigned value
        for child_node in ast.walk(assigned_value):
            # If a function name is detected
            if isinstance(child_node, ast.Name) and child_node.id in self.ast_info.function_information:
                func_call_details = FunctionCallInfo(
                    line_number=assignment_node.lineno,
                    ast_object=assignment_node,
                    object_type="Assign",
                    caller_object=function_node,
                    caller_object_name=function_node.name,
                    callee_object=assigned_value,
                    callee_object_name=child_node.id,
                    assign_targets=assign_targets,
                    call_func_params=assigned_value.args if isinstance(assigned_value, ast.Call) else [],
                )

                # Store the detected function call details for further analysis
                self.ast_info.function_call_info_class[assignment_node] = func_call_details

    def identify_imports_in_function_assignment(
            self, child, assign_target_list, function_info: FunctionDefinition
    ):
        """
        Find import in assign object in function object
        """
        # Extract the value being assigned
        assignment_value = child.value

        # If the value is a function call or binary operation, inspect it for potential imports
        if isinstance(assignment_value, (ast.Call, ast.BinOp)):
            for node_child in ast.walk(assignment_value):
                self._identify_imports_in_assignment(node_child, assign_target_list, function_info)

        # Store the recognized assignment targets in the function information
        function_info.assigned_targets.extend(assign_target_list)

    def _identify_imports_in_assignment(
            self,
            assign_node_value_child,
            assign_targets_list,
            specific_func_info: (NonFunctionInfoClass, FunctionDefinition),
    ):
        """
        Find import list from assign target
        """
        # Early exit if the child node isn't of the 'ast.Name' type
        if not isinstance(assign_node_value_child, ast.Name):
            return

        # Extract the identifier from the child node
        value_id = assign_node_value_child.id

        #  Loop through each assignment target
        for each_assign_target in assign_targets_list:
            # Filter out the matching import information based on the extracted identifier
            matched_imports = [
                import_info
                for import_info in self.ast_info.import_information.values()
                if value_id in import_info.import_as
                   or value_id == import_info.import_name
                   or value_id in import_info.assign_targets_from_non_func
            ]
            # For each matched import, add it to the associated data structures
            for import_info in matched_imports:
                logger.debug(import_info.import_name)
                import_info.assign_targets_from_non_func.add(each_assign_target)
                specific_func_info.imports.add(import_info.import_name)

    def identify_function_calls(self, expression_value, expr_node, func_node):
        """
        Find function call in expr object in function object
        """
        for expr_value_child in ast.walk(expression_value):
            if isinstance(expr_value_child, ast.Name):
                name_id = expr_value_child.id

                func_list = list(self.ast_info.function_information.keys())

                if name_id in func_list:
                    func_call_info_class = FunctionCallInfo(
                        line_number=expr_node.lineno,
                        ast_object=expr_node,
                        object_type="Expr",
                        caller_object=func_node,
                        caller_object_name=func_node.name,
                        callee_object=expression_value,
                        callee_object_name=name_id,
                        call_func_params=expression_value.args,
                    )
                    self.ast_info.function_call_info_class[expr_node] = func_call_info_class

    def find_imports_in_expression(self, expr_value, specific_func_dict: FunctionDefinition):
        """
        Find import in expr object in function object
        """
        for child in ast.walk(expr_value):
            if isinstance(child, ast.Name):
                self.identify_imports_in_name(child, specific_func_dict)

    def identify_imports_in_name(self, name_node, func_info):
        """
        Find import in name object
        """
        node_id = name_node.id

        matching_imports = [
            import_info.import_name
            for import_info in self.ast_info.import_information.values()
            if node_id in (
                import_info.import_as,
                import_info.from_import,
                [import_info.import_name],
                import_info.assign_targets_from_non_func
            )
        ]

        func_info.imports.update(matching_imports)

    def extract_imports_from_expression(
            self, expr_node: ast.Expr, non_func_obj: NonFunctionInfoClass
    ):
        """
        Find import in expr object in non-function object
        """
        # Set the object type of the non-function object to "Expr"
        non_func_obj.object_type = "Expr"
        # Clear any previous assignment details
        non_func_obj.assignments = None
        # Traverse the expression node in search of 'ast.Name' child nodes
        for child in ast.walk(expr_node):
            if isinstance(child, ast.Name):
                # Identify and store potential import names found in the child node
                self.identify_imports_in_name(child, non_func_obj)

    def extract_imports_from_assignment(self, node: (ast.Assign, ast.AugAssign), info: NonFunctionInfoClass, ):
        """
        Extract import details from an assignment node that's outside any function.
        """
        info.object_type = "Assign"
        assignments = extract_assignment_targets(node)

        if isinstance(node.value, (ast.Call, ast.BinOp)):
            for child in ast.walk(node.value):
                self._identify_imports_in_assignment(child, assignments, info)

        info.assignments.extend(assignments)

    def iterate_module_and_analyze(self, module_node: ast.Module):
        """
         Description:
        This method iterates over each child node of the provided AST module. Based on
        the type of the node, it delegates the processing and analysis to specific handlers:

        - If the node represents a function definition (`ast.FunctionDef`), it is sent
          to the `process_function_nodes` method for further analysis specific to functions.

        - Nodes representing module-level imports (`ast.ImportFrom`, `ast.Import`) and
          conditional structures (`ast.If`) are processed using the `process_non_function_nodes` method.

        - In case the node is an `ast.If` type, which potentially checks for the main
          execution of a script (e.g., `if __name__ == "__main__":`), it is sent to the
          `handle_if_statement` method for specialized handling.
        """
        for node in ast.iter_child_nodes(module_node):
            if isinstance(node, ast.FunctionDef):
                self.process_function_nodes(node)
            elif isinstance(node, (ast.ImportFrom, ast.Import)):
                self.process_non_function_nodes(node)
            elif isinstance(node, ast.If):
                self.handle_if_statement(node)

    def process_non_function_nodes(self, node):
        """
        Save non_func_obj_info and find imports in objects
        """
        non_function_info = NonFunctionInfoClass(ast_object=node, line_number=node.lineno)
        self.ast_info.non_function_object_info[node] = non_function_info

        if isinstance(node, (ast.Assign, ast.AugAssign)):
            self.extract_imports_from_assignment(node, non_function_info)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self.extract_imports_from_expression(node, non_function_info)

    def _process_child_node_tree(self, node, parent_function_node, each_func_info):
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Expr):
                self.analyze_expression(child_node, parent_function_node, each_func_info)
            elif isinstance(child_node, (ast.Assign, ast.AugAssign)):
                self.analyze_assign_objects(child_node, parent_function_node, each_func_info)
            elif isinstance(child_node, ast.Name):
                self.identify_imports_in_name(child_node, each_func_info)

    def _analyze_with_items(self, with_node, each_func_info):
        for each_with_item in with_node.items:
            if isinstance(each_with_item, ast.withitem):
                call_obj_in_with = each_with_item.context_expr
                self.find_imports_in_expression(call_obj_in_with, each_func_info)
                self.identify_function_calls(call_obj_in_with, each_with_item, with_node)

    def _handle_child_node(self, child, parent_function_node, each_func_info):
        if isinstance(child, ast.Expr):
            self.analyze_expression(child, parent_function_node, each_func_info)
        elif isinstance(child, (ast.Assign, ast.AugAssign)):
            self.analyze_assign_objects(child, parent_function_node, each_func_info)
        elif isinstance(child, ast.With):
            self._analyze_with_items(child, each_func_info)
        else:
            self._process_child_node_tree(child, parent_function_node, each_func_info)

    def process_function_nodes(self, node: ast.FunctionDef) -> None:
        """
        Find expr and assign in function object and find import and function call
        """
        func_dict = self.ast_info.function_information
        each_func_info: FunctionDefinition = func_dict.get(node.name)
        get_parameters_and_return_objects(node, each_func_info)

        for child in ast.iter_child_nodes(node):
            self._handle_child_node(child, node, each_func_info)

    def handle_if_statement(self, node):
        """
        Handle if __name__ == "__main__":
        """
        if isinstance(node.test, ast.Compare):
            non_func_obj = NonFunctionInfoClass(ast_object=node, line_number=node.lineno)
            self.ast_info.if_main_object = non_func_obj

            if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                for child_node in node.body:
                    if isinstance(child_node, (ast.Assign, ast.AugAssign)):
                        self.find_function_call_of_assign_obj_inside_if_main(child_node)
                    elif isinstance(child_node, ast.Expr) and isinstance(child_node.value, ast.Call):
                        self.find_function_call_of_expr_obj_inside_if_main(child_node)

    def find_function_call_of_assign_obj_inside_if_main(self, child_node):
        """
        Find function call of assign object inside if __name__ == "__main__"
        """
        non_func_obj = NonFunctionInfoClass(ast_object=child_node, line_number=child_node.lineno)
        self.ast_info.objects_inside_if_main[child_node] = non_func_obj

        if isinstance(child_node.value, (ast.Call, ast.BinOp)):
            assign_node_value_obj = child_node.value

            for assign_node_value_child in ast.walk(assign_node_value_obj):
                if isinstance(assign_node_value_child, ast.Name):
                    name_id = assign_node_value_child.id
                    func_list = list(self.ast_info.function_information.keys())
                    if name_id in func_list:
                        func_call_info_class = FunctionCallInsideIfMain(
                            line_number=child_node.lineno,
                            ast_object=child_node,
                            object_type="Assign",
                            callee_object=assign_node_value_obj,
                            callee_object_name=name_id,
                        )
                        self.ast_info.function_call_inside_if_main.setdefault(child_node, []).append(
                            func_call_info_class)

    def find_function_call_of_expr_obj_inside_if_main(self, child_node):
        """
        Find function call of expr object inside if __name__ == "__main__"
        """
        non_func_obj = NonFunctionInfoClass(ast_object=child_node, line_number=child_node.lineno)
        self.ast_info.objects_inside_if_main[child_node] = non_func_obj
        expr_value = child_node.value
        for expr_value_child in ast.walk(expr_value):
            if isinstance(expr_value_child, ast.Name):
                name_id = expr_value_child.id
                func_list = list(self.ast_info.function_information.keys())
                if name_id in func_list:
                    func_call_info_class = FunctionCallInsideIfMain(
                        line_number=child_node.lineno,
                        ast_object=child_node,
                        object_type="Expr",
                        callee_object=expr_value,
                        callee_object_name=name_id,
                    )
                    self.ast_info.function_call_inside_if_main.setdefault(child_node, []).append(
                        func_call_info_class)

    def analyze_assign_objects(self, child, node, specific_func_dict):
        """
        handle assign object in function or non_func_obj
        """
        target_list = extract_assignment_targets(child)
        self.identify_imports_in_function_assignment(child, target_list, specific_func_dict)
        self.find_function_calls_within_assignment(target_list, child, node)

    def analyze_expression(self, expression_node: ast.Expr, parent_node: ast.AST, function_info) -> None:
        """
        Analyze expression nodes to identify imports and function calls, excluding comments.
        """
        if isinstance(expression_node.value, ast.Call):
            self.find_imports_in_expression(expression_node.value, function_info)
            self.identify_function_calls(expression_node.value, expression_node, parent_node)

    def ensure_imports_for_lambda(self, *modules: str) -> None:
        """
        Add boto3 and json imports for Lambda when the code doesn't have them
        """
        existing_imports = set(self.ast_info.import_information.keys())
        for module in modules:
            if module not in existing_imports:
                self.add_module_import(module)

    def add_module_import(self, module_name):
        """
        Add boto3 and json imports for Lambda when the code doesn't have them
        """
        import_node = ast.Import(names=[ast.alias(name=module_name, asname=None)])
        import_node = ast.fix_missing_locations(import_node)
        import_info = ModuleImportClass(ast_object=import_node)
        self.ast_info.boto3_and_json_imports[module_name] = import_info

    def start_analyzing(self):
        """
        Find import that is python code
        Iterate module and analyze all for import statements and functions
        Lastly add boto3 and json imports
        """

        self.find_imports_from_python_scripts(self.file_name)
        self.iterate_module_and_analyze(self.target_source_code)
        self.ensure_imports_for_lambda("boto3", "json")


def get_parameters_and_return_objects(node, specific_func_dict):
    """
    find parameters and return information in function object

    Parameters
    ----------
    node : ast.FunctionDef
    specific_func_dict : FunctionDefinition
    """
    function_parameters = [
        VALUE
        for obj in node.args.args
        for fieldname, VALUE in ast.iter_fields(obj)
        if fieldname == "arg"
    ]
    specific_func_dict.parameters = function_parameters

    # last object
    last_object_in_function = node.body[-1]
    if isinstance(last_object_in_function, ast.Return):

        specific_func_dict.return_values_ast = last_object_in_function

        # when number of return object is more than 2
        if isinstance(last_object_in_function.value, ast.Tuple):
            return_objects = list(last_object_in_function.value.elts)
            specific_func_dict.return_values = return_objects

        elif isinstance(last_object_in_function.value, ast.Name):
            specific_func_dict.return_values.append(last_object_in_function.value)


def _process_assign_targets(targets: List[ast.expr]) -> List[str]:
    """
    Process assignment targets for the ast.Assign node.
    """
    result = []
    for target in targets:
        if isinstance(target, ast.Tuple):
            result.extend(el.id for el in target.elts if isinstance(el, ast.Name))
        elif isinstance(target, ast.Subscript):
            result.append(astor.to_source(target).strip())
        elif isinstance(target, ast.Name):
            result.append(target.id)
    return result


def _process_aug_assign_target(target: ast.expr) -> List[str]:
    """
    Process assignment target for the ast.AugAssign node.
    """
    if isinstance(target, ast.Tuple):
        return [el.id for el in target.elts if isinstance(el, ast.Name)]
    elif isinstance(target, ast.Name):
        return [target.id]
    return []


def extract_assignment_targets(node: (ast.Assign, ast.AugAssign)):
    """
    Get Assign Target for example a = 3 -> get a,    b += 3 -> get b
    """

    if isinstance(node, ast.Assign):
        return _process_assign_targets(node.targets)
    elif isinstance(node, ast.AugAssign):
        return _process_aug_assign_target(node.target)
    return []


def transform_to_lambda_functions(whole_ast_info: WholeASTInfoClass):
    """
    Make function lambda based on annotation information
    """
    logger.info("Changing function definitions to lambda functions")

    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app]: Creating a single function")
        convert_main_function_to_lambda(whole_ast_info)

    function_information = whole_ast_info.function_information
    sort_by_lambda_group = whole_ast_info.sort_by_lambda_group
    lambda_counter = 1

    for lambda_group, functions in sort_by_lambda_group.items():
        if lambda_group == "Default":
            lambda_counter = transform_default_group_to_lambda(
                function_information,
                functions,
                lambda_counter,
                whole_ast_info,
            )
        else:
            lambda_counter = transform_group_to_lambda(
                function_information,
                functions,
                lambda_counter,
                lambda_group,
                whole_ast_info,
            )


def transform_default_group_to_lambda(
        function_information, function_list, lambda_number_idx, whole_ast_info
):
    """
    Make lambda function for default group
    """

    for each_func_name in function_list:
        original_func_info = function_information.get(each_func_name)

        # Change lambda function name
        func_name = f"lambda_handler_{str(lambda_number_idx)}"
        function_args = get_default_lambda_function_inputs()

        func_call_arguments_list = []
        function_body = []
        function_event_input_assign_list = []
        return_objects = []

        for each_param in original_func_info.parameters:
            call_args = ast.Name(id=each_param, ctx=ast.Store())
            func_call_arguments_list.append(call_args)

            subscript_ast = ast.Subscript(
                value=ast.Name(id="event", ctx=ast.Load()),
                slice=ast.Index(value=ast.Str(s=each_param)),
                ctx=ast.Load(),
            )
            name_ast = ast.Name(id=each_param, ctx=ast.Store())
            assign_ast = ast.Assign(targets=[name_ast], value=subscript_ast)

            function_body.append(assign_ast)
            function_event_input_assign_list.append(assign_ast)

        value_obj = ast.Call(
            func=ast.Name(id=original_func_info.func_name, ctx=ast.Load()),
            args=func_call_arguments_list,
            keywords=[],
        )

        if original_func_info.return_values:
            assign_target_obj = ast.Name(id="result", ctx=ast.Store())
            assign_ast = ast.Assign(targets=[assign_target_obj], value=value_obj)
            function_body.append(assign_ast)
            return_obj = ast.Return(value=ast.Name(id="result", ctx=ast.Load()))
            function_body.append(return_obj)
            return_objects.append(return_obj)
        else:
            expr_obj = ast.Expr(value=value_obj)
            function_body.append(expr_obj)

        lambda_handler = ast.FunctionDef(
            name=func_name,
            args=function_args,
            body=function_body,
            decorator_list=[],
            returns=None,
        )

        lambda_module = ast.Module(
            body=[original_func_info.ast_object] + [lambda_handler]
        )

        compiler_generated_lambda_info = CompilerGeneratedLambda(
            lambda_name=func_name,
            original_ast_object=original_func_info.ast_object,
            lambda_module=lambda_module,
            lambda_handler_func_object=lambda_handler,
            original_func_name=original_func_info.func_name,
            lambda_event_input_objs=function_event_input_assign_list,
            input_parameter=original_func_info.parameters,
            return_objects=return_objects,
        )

        whole_ast_info.lambda_function_info[
            original_func_info.func_name
        ] = compiler_generated_lambda_info

        lambda_number_idx += 1

    return lambda_number_idx


def transform_group_to_lambda(
        function_information, function_list, lambda_number_idx, lambda_group, whole_ast_info
):
    """
    Make lambda function for lambda group
    """
    func_info_list = [function_information.get(x) for x in function_list]
    sorted_func_info_list = sorted(func_info_list, key=lambda x: x.line_number)

    subscript_ast = ast.Subscript(
        value=ast.Name(id="event", ctx=ast.Load()),
        slice=ast.Index(value=ast.Str(s="func_with_params")),
        ctx=ast.Load(),
    )
    name_ast = ast.Name(id="func_with_params", ctx=ast.Store())
    func_with_params_assign_ast = ast.Assign(targets=[name_ast], value=subscript_ast)

    lambda_event_inputs_per_function = defaultdict(list)
    for func_info in sorted_func_info_list:
        make_func_call_for_grouped_lambda_handler(
            func_info, lambda_event_inputs_per_function
        )

    lambda_combination_list = []
    for i in range(len(sorted_func_info_list)):
        lambda_combination_list.extend(
            iter(itertools.combinations(sorted_func_info_list, i + 1))
        )

    if_statement_list = []
    lambda_input_per_if_statement = defaultdict()
    parse_input_per_if_statement = defaultdict()
    return_obj_per_if_statement = defaultdict()

    for each_comb in lambda_combination_list:
        if len(each_comb) == 1:
            make_func_call_for_one_func_combination(
                each_comb,
                if_statement_list,
                lambda_event_inputs_per_function,
                lambda_input_per_if_statement,
                parse_input_per_if_statement,
                return_obj_per_if_statement,
            )
        else:
            make_func_call_for_multiple_funcs_combination(
                each_comb,
                if_statement_list,
                lambda_event_inputs_per_function,
                lambda_input_per_if_statement,
                parse_input_per_if_statement,
                return_obj_per_if_statement,
            )

    lambda_handler = ast.FunctionDef(
        name=f"lambda_handler_{str(lambda_number_idx)}",
        args=get_default_lambda_function_inputs(),
        body=[func_with_params_assign_ast] + if_statement_list,
        decorator_list=[],
        returns=None,
    )

    function_object_list = [x.ast_object for x in sorted_func_info_list]
    lambda_module = ast.Module(body=function_object_list + [lambda_handler])
    ast_object_list = [x.ast_object for x in sorted_func_info_list]

    merged_lambda_info = MergedCompilerGeneratedLambda(
        lambda_group_name=f"lambda_handler_{str(lambda_number_idx)}",
        lambda_name_list=function_list,
        lambda_module=lambda_module,
        lambda_handler_func_object=lambda_handler,
        original_ast_object_list=ast_object_list,
        lambda_event_input_objs=lambda_event_inputs_per_function,
        lambda_input_per_if_statement=lambda_input_per_if_statement,
        parse_input_per_if_statement=parse_input_per_if_statement,
        return_obj_per_if_statement=return_obj_per_if_statement,
    )
    whole_ast_info.lambda_function_info[lambda_group] = merged_lambda_info
    lambda_number_idx += 1

    return lambda_number_idx


def make_func_call_for_one_func_combination(
        each_comb,
        if_statement_list,
        lambda_event_inputs_per_function,
        input_per_if_statement,
        parse_input_per_if_statement,
        return_obj_per_if_statement,
):
    """
    Make function call for one function combination
    """
    func_info = each_comb[0]

    lambda_event_input_list = []
    event_input_arg_obj_list = []
    for each_argument in lambda_event_inputs_per_function[func_info.func_name]:
        name_ast = ast.Name(id=each_argument.slice.value.s, ctx=ast.Store())
        assign_ast = ast.Assign(targets=[name_ast], value=each_argument)
        event_input_arg_obj_list.append(assign_ast)
        lambda_event_input_list.append(name_ast)

    left_obj = ast.Name(id="func_with_params", ctx=ast.Load())
    ops_obj = [ast.Eq()]
    comparators_obj = [ast.Str(s=func_info.func_name)]
    test_obj = ast.Compare(left=left_obj, ops=ops_obj, comparators=comparators_obj)

    return_objs = []
    if func_info.return_values:
        return_obj = ast.Return(value=ast.Name(id="result", ctx=ast.Load()))
        return_objs.append(return_obj.value.id)
        if_statement_obj = ast.If(
            test=test_obj,
            body=event_input_arg_obj_list + [func_info.lambda_handler_call] + [return_obj],
            orelse=[],
        )
    else:
        if_statement_obj = ast.If(
            test=test_obj,
            body=event_input_arg_obj_list + [func_info.lambda_handler_call],
            orelse=[],
        )

    if_statement_list.append(if_statement_obj)
    input_per_if_statement[if_statement_obj] = lambda_event_input_list
    parse_input_per_if_statement[if_statement_obj] = event_input_arg_obj_list
    return_obj_per_if_statement[if_statement_obj] = return_objs


def make_func_call_for_multiple_funcs_combination(
        function_combination_list,
        if_statement_list,
        lambda_event_inputs_per_function,
        input_per_if_statement,
        parse_input_per_if_statement,
        return_obj_per_if_statement,
):
    """
    Make function call for multiple functions combination
    """
    lambda_name_list = [func_info.func_name for func_info in function_combination_list]
    combined_function_name = "_and_".join(lambda_name_list)

    first_function_info = function_combination_list[0]

    if isinstance(first_function_info.lambda_handler_call, ast.Assign):
        lambda_event_input_list = []
        event_input_arg_obj_list = []
        for each_argument in lambda_event_inputs_per_function[first_function_info.func_name]:
            name_ast = ast.Name(id=each_argument.slice.value.s, ctx=ast.Store())
            assign_ast = ast.Assign(targets=[name_ast], value=each_argument)
            event_input_arg_obj_list.append(assign_ast)
            lambda_event_input_list.append(name_ast)

        assign_targets = first_function_info.lambda_handler_call.targets

        remaining_func_call_except_first = [
            x.lambda_handler_call for x in function_combination_list[1:]
        ]

        copied_remaining_func_call_except_first = copy.deepcopy(
            remaining_func_call_except_first
        )
        for x in copied_remaining_func_call_except_first:
            x.value.args = assign_targets

        left_obj = ast.Name(id="func_with_params", ctx=ast.Load())
        ops_obj = [ast.Eq()]
        comparators_obj = [ast.Str(s=combined_function_name)]
        test_obj = ast.Compare(left=left_obj, ops=ops_obj, comparators=comparators_obj)

        last_func_info = function_combination_list[-1]
        return_objs = []
        if last_func_info.return_values:
            return_obj = ast.Return(value=ast.Name(id="result", ctx=ast.Load()))
            return_objs.append(return_obj.value.id)
            if_statement_obj = ast.If(
                test=test_obj,
                body=event_input_arg_obj_list
                     + [first_function_info.lambda_handler_call]
                     + copied_remaining_func_call_except_first
                     + [return_obj],
                orelse=[],
            )
        else:
            if_statement_obj = ast.If(
                test=test_obj,
                body=event_input_arg_obj_list
                     + [first_function_info.lambda_handler_call]
                     + copied_remaining_func_call_except_first,
                orelse=[],
            )

        if_statement_list.append(if_statement_obj)
        input_per_if_statement[if_statement_obj] = lambda_event_input_list
        parse_input_per_if_statement[if_statement_obj] = event_input_arg_obj_list
        return_obj_per_if_statement[if_statement_obj] = return_objs


def make_func_call_for_grouped_lambda_handler(func_info, lambda_event_inputs_per_function):
    """
    Make function call for grouped lambda handler
    """
    func_call_arguments_list = []

    for each_param in func_info.parameters:
        subscript_ast = ast.Subscript(
            value=ast.Subscript(
                ast.Name(id="event", ctx=ast.Load()),
                slice=ast.Index(value=ast.Str(s=func_info.func_name)),
                ctx=ast.Load(),
            ),
            slice=ast.Index(value=ast.Str(s=each_param)),
            ctx=ast.Load(),
        )

        call_args = ast.Name(id=each_param, ctx=ast.Store())
        func_call_arguments_list.append(call_args)
        lambda_event_inputs_per_function[func_info.func_name].append(subscript_ast)

    value_obj = ast.Call(
        func=ast.Name(id=func_info.func_name, ctx=ast.Load()),
        args=func_call_arguments_list,
        keywords=[],
    )
    if func_info.return_values:
        assign_target_obj = ast.Name(id="result", ctx=ast.Store())
        assign_ast = ast.Assign(targets=[assign_target_obj], value=value_obj)
        func_info.lambda_handler_call = assign_ast
    else:
        expr_obj = ast.Expr(value=value_obj)
        func_info.lambda_handler_call = expr_obj


def make_lambda_function_for_default_group(
        each_func_name, function_information, lambda_number_idx, whole_ast_info
):
    # Get function info class for each function
    original_func_info = function_information.get(each_func_name)

    # Change lambda function name
    func_name = f"lambda_handler_{str(lambda_number_idx)}"
    function_args = get_default_lambda_function_inputs()

    # Get arguments for function parameters with a = event['a']
    func_call_arguments_list = []  # for function callee
    function_body = []  # a = event['a]
    function_event_input_assign_list = []  # a, b, c
    return_objects = []  # return result

    for each_param in original_func_info.parameters:
        # func_call(A,B) -> get A, B
        call_args = ast.Name(id=each_param, ctx=ast.Store())
        func_call_arguments_list.append(call_args)

        # A = event['A'], B = event['B']
        subscript_ast = ast.Subscript(
            value=ast.Name(id="event", ctx=ast.Load()),
            slice=ast.Index(value=ast.Str(s=each_param)),
            ctx=ast.Load(),
        )
        name_ast = ast.Name(id=each_param, ctx=ast.Store())
        assign_ast = ast.Assign(targets=[name_ast], value=subscript_ast)

        # Add A = event['A']
        function_body.append(assign_ast)  # a = event['a']

        # For saving it to lambda function info
        function_event_input_assign_list.append(assign_ast)

    # function callee in lambda handler function
    value_obj = ast.Call(
        func=ast.Name(id=original_func_info.func_name, ctx=ast.Load()),
        args=func_call_arguments_list,
        keywords=[],
    )

    # If return object exist
    if original_func_info.return_values:

        # Add assign object with main function callee
        assign_target_obj = ast.Name(id="result", ctx=ast.Store())
        assign_ast = ast.Assign(targets=[assign_target_obj], value=value_obj)
        function_body.append(assign_ast)

        # Append return object
        return_obj = ast.Return(value=ast.Name(id="result", ctx=ast.Load()))
        function_body.append(return_obj)
        return_objects.append(return_obj)
    else:

        # Just add function call with no assigned targets
        expr_obj = ast.Expr(value=value_obj)
        function_body.append(expr_obj)

    # Create new lambda handler with main function callee_object
    lambda_handler = ast.FunctionDef(
        name=func_name,
        args=function_args,
        body=function_body,
        decorator_list=[],
        returns=None,
    )

    # Add function definition and lambda handler
    lambda_module = ast.Module(body=[original_func_info.ast_object] + [lambda_handler])

    # Save it into whole_information
    compiler_generated_lambda_info = CompilerGeneratedLambda(
        lambda_name=func_name,
        original_ast_object=original_func_info.ast_object,
        lambda_module=lambda_module,
        lambda_handler_func_object=lambda_handler,
        original_func_name=original_func_info.func_name,
        lambda_event_input_objs=function_event_input_assign_list,
        input_parameter=original_func_info.parameters,
        return_objects=return_objects,
    )
    whole_ast_info.lambda_function_info[
        original_func_info.func_name
    ] = compiler_generated_lambda_info


def _create_lambda_argument_assignments(main_function_info):
    """
    Create assignments and function call arguments for the lambda function based on main function parameters.

    Returns:
    Tuple[List[ast.Name], List[ast.Assign]]
    """
    assignments = []
    func_call_arguments = []

    for each_param in main_function_info.parameters:
        subscript_ast = ast.Subscript(
            value=ast.Name(id="event", ctx=ast.Load()),
            slice=ast.Index(value=ast.Str(s=each_param)),
            ctx=ast.Load(),
        )
        name_ast = ast.Name(id=each_param, ctx=ast.Store())
        assign_ast = ast.Assign(targets=[name_ast], value=subscript_ast)

        func_call_arguments.append(name_ast)
        assignments.append(assign_ast)

    return func_call_arguments, assignments


def _handle_return_values(main_function_info, value_obj, assignments):
    """
    Handle return values of the main function for lambda conversion.

    Returns:
    Tuple[List[Union[ast.Assign, ast.Expr, ast.Return]], List[ast.Return]]
    """
    function_body = assignments.copy()
    return_objects = []

    if main_function_info.return_values:
        assign_target_obj = ast.Name(id="result", ctx=ast.Store())
        assign_ast = ast.Assign(targets=[assign_target_obj], value=value_obj)
        function_body.extend([assign_ast, ast.Return(value=ast.Name(id="result", ctx=ast.Load()))])
        return_objects.append(assign_ast)
    else:
        function_body.append(ast.Expr(value=value_obj))

    return function_body, return_objects


def convert_main_function_to_lambda(whole_ast_info: WholeASTInfoClass):
    # Fetch main function definition
    main_function_info = whole_ast_info.function_information.get("main")

    # Generate the lambda function name
    func_name = f"lambda_handler_{whole_ast_info.file_name.split('.')[0]}"
    function_args = get_default_lambda_function_inputs()

    # Extract arguments and create assignments based on the function parameters
    func_call_arguments, assignments = _create_lambda_argument_assignments(main_function_info)

    # Construct the function call for the main function inside the lambda handler
    value_obj = ast.Call(
        func=ast.Name(id=main_function_info.func_name, ctx=ast.Load()),
        args=func_call_arguments,
        keywords=[],
    )

    # Handle return values if any
    function_body, return_objects = _handle_return_values(main_function_info, value_obj, assignments)

    # Create new lambda handler with main function callee_object
    lambda_handler = ast.FunctionDef(
        name=func_name,
        args=function_args,
        body=function_body,
        decorator_list=[],
        returns=None,
    )

    # Update the module body to incorporate the new lambda handler
    module_body = whole_ast_info.copied_module_for_analysis.body[:]
    module_body.remove(whole_ast_info.if_main_object.ast_object)
    lambda_module = ast.Module(body=module_body + [lambda_handler])

    # Store the information related to the generated lambda
    compiler_generated_lambda_info = CompilerGeneratedLambda(
        lambda_name=func_name,
        original_ast_object=main_function_info.ast_object,
        lambda_module=lambda_module,
        lambda_handler_func_object=lambda_handler,
        original_func_name=main_function_info.func_name,
        lambda_event_input_objs=assignments,
        input_parameter=main_function_info.parameters,
        return_objects=return_objects,
    )
    whole_ast_info.module_info_for_offloading_whole_app = compiler_generated_lambda_info


def get_default_lambda_function_inputs():
    return ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg="event", annotation=None),
            ast.arg(arg="context", annotation=None),
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def make_if_block_for_invoke_function() -> ast.If:
    """
    Construct the if block to determine the appropriate function invocation
    based on the type of object.
    """
    invoke_for_assign = [
        invoke_func_for_assign_obj(),
        parse_result_from_lambda_result(),
        download_output_to_vm(),
        ast.Return(value=ast.Name(id="result", ctx=ast.Load())),
    ]

    invoke_for_expr = [invoke_func_for_expr_obj()]

    return ast.If(
        test=ast.Name(id="assign_obj", ctx=ast.Load()),
        body=invoke_for_assign,
        orelse=[
            ast.If(
                test=ast.Name(id="expr_obj", ctx=ast.Load()),
                body=invoke_for_expr,
                orelse=[],
            ),
        ],
    )


def make_orchestrator_function() -> ast.FunctionDef:
    """
    Make orchestrator function for invoking lambda function
    """
    function_name = "invoke_function_using_lambda"
    function_args_field_obj = return_function_args_object()
    upload_input_obj = make_for_object_of_uploading_input_to_s3()
    invoke_function_block = make_if_block_for_invoke_function()

    function_body = [upload_input_obj, invoke_function_block]

    return ast.FunctionDef(
        name=function_name,
        args=function_args_field_obj,
        body=function_body,
        decorator_list=[],
        returns=None,
    )


def invoke_func_for_expr_obj():
    return ast.Expr(value=make_lambda_invoke_using_client_object())


def make_lambda_invoke_using_client_object():
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="lambda_client", ctx=ast.Load()),
            attr="invoke",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[
            ast.keyword(
                arg="FunctionName",
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="map_func_to_func_arn_dict", ctx=ast.Load()),
                        attr="get",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id="function_name", ctx=ast.Load()), ],
                    keywords=[],
                ),
            ),
            ast.keyword(arg="InvocationType", value=ast.Str(s="RequestResponse")),
            ast.keyword(
                arg="Payload",
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="json", ctx=ast.Load()),
                        attr="dumps",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id="input_dict", ctx=ast.Load()), ],
                    keywords=[],
                ),
            ),
        ],
    )


def download_output_to_vm():
    return ast.If(
        test=ast.Name(id="download_output_from_s3", ctx=ast.Load()),
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="s3_client", ctx=ast.Load()),
                        attr="download_file",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Str(s="bucket-for-microblend-compiler"),
                        ast.Name(id="result", ctx=ast.Load()),
                        ast.BinOp(
                            left=ast.Str(s="/tmp/"),
                            op=ast.Add(),
                            right=ast.Name(id="result", ctx=ast.Load()),
                        ),
                    ],
                    keywords=[],
                )
            ),
        ],
        orelse=[],
    )


def parse_result_from_lambda_result():
    return ast.Assign(
        targets=[ast.Name(id="result", ctx=ast.Store()), ],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="json", ctx=ast.Load()), attr="loads", ctx=ast.Load()
            ),
            args=[
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Subscript(
                                    value=ast.Name(id="response", ctx=ast.Load()),
                                    slice=ast.Index(value=ast.Str(s="Payload")),
                                    ctx=ast.Load(),
                                ),
                                attr="read",
                                ctx=ast.Load(),
                            ),
                            args=[],
                            keywords=[],
                        ),
                        attr="decode",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[],
                ),
            ],
            keywords=[],
        ),
    )


def invoke_func_for_assign_obj():
    return ast.Assign(
        targets=[ast.Name(id="response", ctx=ast.Store()), ],
        value=make_lambda_invoke_using_client_object(),
    )


def return_function_args_object() -> ast.arguments:
    """
    Return ast arguments object with specified function parameters and defaults.
    """

    # Define parameters without defaults
    parameters_without_defaults = ["function_name", "input_dict"]

    # Define parameters with defaults
    parameters_with_defaults = [
        "assign_obj",
        "expr_obj",
        "input_to_s3",
        "download_output_from_s3",
    ]

    # Generate ast.arg objects for both lists
    args_without_defaults = [ast.arg(arg=param, annotation=None) for param in parameters_without_defaults]
    args_with_defaults = [ast.arg(arg=param, annotation=None) for param in parameters_with_defaults]

    # Create a list of default values for args with defaults
    default_values = [ast.NameConstant(value=False) for _ in parameters_with_defaults]

    # Combine arguments and defaults to create the ast.arguments object
    return ast.arguments(
        args=args_without_defaults + args_with_defaults,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=default_values,
    )


def make_for_object_of_uploading_input_to_s3():
    return ast.For(
        target=ast.Name(id="each_input", ctx=ast.Store()),
        iter=return_iteration_for_uploading_input_to_s3(),
        body=[return_body_object_for_uploading_input_to_s3()],
        orelse=[],
    )


def return_body_object_for_uploading_input_to_s3():
    return ast.If(
        test=ast.Name(id="input_to_s3", ctx=ast.Load()),
        body=[
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="s3_client", ctx=ast.Load()),
                        attr="upload_file",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.BinOp(
                            left=ast.Str(s="/tmp/"),
                            op=ast.Add(),
                            right=ast.Name(id="each_input", ctx=ast.Load()),
                        ),
                        ast.Str(s="bucket-for-microblend-compiler"),
                        ast.Name(id="each_input", ctx=ast.Load()),
                    ],
                    keywords=[],
                )
            ),
        ],
        orelse=[],
    )


def return_iteration_for_uploading_input_to_s3():
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="input_dict", ctx=ast.Load()),
            attr="values",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[],
    )


def return_func_call_arguments(function_return_objects) -> List[ast.keyword]:
    """
    Fetch download_output_from_s3, assign_obj, input_to_s3
    """

    if function_return_objects:
        ast_keyword_for_download_output_from_s3 = ast.keyword(
            arg="download_output_from_s3", value=ast.NameConstant(value=True)
        )

        return [
            ast.keyword(arg="assign_obj", value=ast.NameConstant(value=True)),
            ast.keyword(arg="input_to_s3", value=ast.NameConstant(value=False)),
            ast_keyword_for_download_output_from_s3,
        ]
    else:
        ast_keyword_for_download_output_from_s3 = ast.keyword(
            arg="download_output_from_s3", value=ast.NameConstant(value=False)
        )

        return [
            ast.keyword(arg="expr_obj", value=ast.NameConstant(value=True)),
            ast.keyword(arg="input_to_s3", value=ast.NameConstant(value=False)),
            ast_keyword_for_download_output_from_s3,
        ]


def change_func_call_for_default_lambda_group(func_call_obj, func_call_obj_name, whole_ast_info):
    # Fetch function information
    function_info_class: FunctionDefinition = whole_ast_info.function_information[func_call_obj_name]
    function_return_objects = function_info_class.return_values

    # Change function name to invoke_func_by_lambda
    # resize (a) -> invoke_function_using_lambda(a)
    func_call_obj.func = ast.Name(id="invoke_function_using_lambda", ctx=ast.Load())

    # Make dictionary for function parameters - {'a' : a},
    # invoke_function_using_lambda('resize', {'img_file_name':a)
    str_obj_of_call_func_params, name_obj_of_call_func_params = [], []
    for each_argument in func_call_obj.args:
        name_obj_of_call_func_params.append(each_argument)
        str_obj_of_call_func_params.append(each_argument.id)

    # add function parameters - download_output_from_s3, input_to_s3, assign_obj
    func_call_obj.keywords = return_func_call_arguments(function_return_objects)

    logger.debug(astor.to_source(func_call_obj))


def check_data_dependency(copied_func_call: FunctionCallInfo, whole_ast_info: WholeASTInfoClass):
    """
    For function call -> b
    a = b()
    c = d(a)
    Then we mark output_to_s3 = True
    """

    input_dependency(copied_func_call, whole_ast_info)
    output_dependency(copied_func_call, whole_ast_info)


def output_dependency(copied_func_call, whole_ast_info):
    """
    For function call
    """
    callee_func_name: str = copied_func_call.caller_object_name
    assign_node_target_list: list = copied_func_call.assigned_targets
    call_obj: ast.Call = copied_func_call.callee_object

    # Iterate function calls to find dependency
    each_function_call: FunctionCallInfo
    for _, each_function_call in whole_ast_info.function_call_info_class.items():

        if callee_func_name == each_function_call.caller_object_name:
            # get function call input parameters and call object
            function_call_input_list = [i.id for i in each_function_call.call_func_params]

            if set(assign_node_target_list).intersection(set(function_call_input_list)):
                # fetch func call object
                calling_obj_from_each_function_call = each_function_call.callee_object

                if call_obj.lineno < calling_obj_from_each_function_call.lineno:
                    for each_keyword in call_obj.keywords:
                        if each_keyword.arg == "download_output_from_s3":
                            each_keyword.value.value = True


def should_update_input_to_s3(each_function_call, callee_func_name, call_func_parameters_name, call_obj):
    """Determine if the input_to_s3 keyword should be updated."""
    if callee_func_name != each_function_call.caller_object_name or each_function_call.object_type != "Assign":
        return False

    assign_targets = each_function_call.assigned_targets
    if not set(assign_targets).intersection(set(call_func_parameters_name)):
        return False

    calling_obj_from_each_function_call = each_function_call.callee_object
    return call_obj.lineno > calling_obj_from_each_function_call.lineno


def update_keyword_value(call_obj, keyword_arg, value):
    """Update a specific keyword's value in the given ast.Call object."""
    for each_keyword in call_obj.keywords:
        if each_keyword.arg == keyword_arg:
            each_keyword.value.value = value


def input_dependency(copied_func_call, whole_ast_info):
    """
    Input to S3 : True
    :param copied_func_call: The copied function call object
    :param whole_ast_info: The information about the whole AST
    """
    call_func_params = copied_func_call.call_func_params
    call_func_parameters_name = [i.id for i in call_func_params]
    callee_func_name: str = copied_func_call.caller_object_name
    call_obj: ast.Call = copied_func_call.callee_object

    for _, each_function_call in whole_ast_info.function_call_info_class.items():
        if should_update_input_to_s3(each_function_call, callee_func_name, call_func_parameters_name, call_obj):
            update_keyword_value(call_obj, "input_to_s3", True)


def function_call_orchestrator(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Convert function calls to lambda function calls.
    """
    logger.info("Make function call orchestrator")

    if not whole_ast_info.offloading_whole_application:
        orchestrate_function_call(whole_ast_info)
    else:
        logger.info("[Offloading whole app] : Skipping function_call_orchestrator")


def get_func_call_of_lambda(func_call, function_of_lambda):
    return sorted(
        [func_call_val for func_call_val in func_call.values() if
         func_call_val.callee_object_name in function_of_lambda],
        key=lambda x: x.line_number
    )


def get_function_of_lambda(sort_by_lambda_group):
    return [func for func_list in sort_by_lambda_group.values() for func in func_list]


def group_by_lambda_calls(func_call_of_lambda, sort_by_lambda_group):
    group_func_call = defaultdict(list)
    for func_call_val in func_call_of_lambda:
        for group_name, func_list in sort_by_lambda_group.items():
            if func_call_val.callee_object_name in func_list:
                group_func_call[group_name].append(func_call_val)
    return group_func_call


def orchestrate_default_group(each_func_call_info, whole_ast_info):
    # Copy function_call_info_class. Need original later for dependency
    copied_func_call = copy.deepcopy(each_func_call_info)
    func_call_obj: ast.Call = copied_func_call.callee_object
    func_call_obj_name = copied_func_call.callee_object_name

    # Add it to whole_information
    whole_ast_info.func_call_using_lambda.append(
        LambdaBasedFunctionCallInfoClass(
            copied_func_call_info=copied_func_call,
            original_func_call_info=each_func_call_info,
        )
    )

    change_func_call_for_default_lambda_group(func_call_obj, func_call_obj_name, whole_ast_info)
    check_data_dependency(copied_func_call, whole_ast_info)


def prepare_lambda_call_objects(first_func_call, func_name_list, function_info_class):
    # Copy function_call_info_class. Need original later for dependency
    copied_func_call = copy.deepcopy(first_func_call)
    func_call_obj: ast.Call = copied_func_call.callee_object

    # Change function name to invoke_func_by_lambda
    func_call_obj.func = ast.Name(id="invoke_function_using_lambda", ctx=ast.Load())

    function_parameters = [ast.Str(s=each_param) for each_param in function_info_class.parameters]

    str_obj_of_call_func_params, name_obj_of_call_func_params = [], []
    for each_argument in func_call_obj.args:
        str_obj_of_call_func_params.append(each_argument.id)
        name_obj_of_call_func_params.append(each_argument)

    func_call_obj.args = [
        ast.Str(s="_and_".join(func_name_list)),
        ast.Dict(
            keys=[ast.Str(s="func_with_params")],
            values=[
                ast.Dict(
                    keys=[ast.Str(s="_and_".join(func_name_list))],
                    values=[ast.Dict(keys=function_parameters, values=name_obj_of_call_func_params)]
                )
            ]
        )
    ]

    func_call_obj.keywords = return_func_call_arguments(function_info_class.return_values)

    return copied_func_call, func_call_obj


def should_update_keyword(each_function_call, callee_func_name, assign_node_target_list, call_obj):
    function_call_input_list = [i.id for i in each_function_call.call_func_params]
    calling_obj_from_each_function_call = each_function_call.callee_object

    conditions = [
        callee_func_name == each_function_call.caller_object_name,
        bool(set(assign_node_target_list).intersection(set(function_call_input_list))),
        call_obj.lineno < calling_obj_from_each_function_call.lineno
    ]

    return all(conditions)


def update_keyword_value(call_obj, keyword_arg, value):
    for each_keyword in call_obj.keywords:
        if each_keyword.arg == keyword_arg:
            each_keyword.value.value = value


def update_keywords_for_download(call_obj, callee_func_name, assign_node_target_list, function_call_info_class):
    for _, each_function_call in function_call_info_class.items():
        if should_update_keyword(each_function_call, callee_func_name, assign_node_target_list, call_obj):
            update_keyword_value(call_obj, "download_output_from_s3", True)
            break
    else:
        update_keyword_value(call_obj, "download_output_from_s3", False)


def merge_function_calls(whole_ast_info, func_name_list, func_call_list, lambda_group):
    logger.info("Merge Function Call")

    first_func_call = func_call_list[0]
    function_info_class: FunctionDefinition = whole_ast_info.function_information[first_func_call.callee_object_name]
    copied_func_call, func_call_obj = prepare_lambda_call_objects(first_func_call, func_name_list, function_info_class)

    first_lambda_info_for_function_call = LambdaBasedFunctionCallInfoClass(
        copied_func_call_info=copied_func_call, original_func_call_info=first_func_call
    )

    last_func_call = func_call_list[-1]
    copied_last_func_call = copy.deepcopy(last_func_call)
    last_lambda_info_for_function_call = LambdaBasedFunctionCallInfoClass(
        copied_func_call_info=None, original_func_call_info=last_func_call
    )

    lambda_info_list = [first_lambda_info_for_function_call, last_lambda_info_for_function_call]
    whole_ast_info.combined_func_call_using_lambda[lambda_group] = lambda_info_list

    input_dependency(copied_func_call, whole_ast_info)

    callee_func_name: str = copied_last_func_call.caller_object_name
    assign_node_target_list: list = copied_last_func_call.assigned_targets
    call_obj: ast.Call = copied_func_call.callee_object

    update_keywords_for_download(call_obj, callee_func_name, assign_node_target_list,
                                 whole_ast_info.function_call_info_class)


def orchestrate_function_call(whole_ast_info):
    """
    Orchestrate function call
    """
    function_information = whole_ast_info.function_information
    sort_by_lambda_group = whole_ast_info.sort_by_lambda_group

    function_of_lambda = get_function_of_lambda(sort_by_lambda_group)
    func_call = whole_ast_info.function_call_info_class
    func_call_of_lambda = get_func_call_of_lambda(func_call, function_of_lambda)
    group_func_call = group_by_lambda_calls(func_call_of_lambda, sort_by_lambda_group)

    # Make lambda call orchestrator function
    if func_call_of_lambda:
        make_function_orchestrator_func(whole_ast_info)

    # For lambda group
    # Sort function definition by line number
    func_info_list = [function_information.get(x) for x in function_of_lambda]
    sorted_func_info_list = sorted(func_info_list, key=lambda x: x.line_number)
    sorted_func_name_list = [x.func_name for x in sorted_func_info_list]

    for lambda_group, func_call_list in group_func_call.items():
        if lambda_group == "Default":
            for each_func_call_info in func_call_list:
                orchestrate_default_group(each_func_call_info, whole_ast_info)

        else:
            merge_function_calls(whole_ast_info, sorted_func_name_list, func_call_list, lambda_group)


def make_function_orchestrator_func(whole_ast_info):
    """
    Make function orchestrator function
    """
    # Make lambda invoke function depending on the flag
    lambda_invoke_func_obj = make_orchestrator_function()
    whole_ast_info.lambda_invoke_function = lambda_invoke_func_obj


def make_obj_for_uploading_return_obj_to_s3(return_object_list):
    """
    s3_client.upload("Return")
    """
    upload_obj_list = []

    for each_input_param in return_object_list:
        upload_obj = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="s3_client", ctx=ast.Load()),
                    attr="upload_file",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.BinOp(
                        left=ast.Str(s="/tmp/"),
                        op=ast.Add(),
                        right=ast.Name(id=each_input_param, ctx=ast.Load()),
                    ),
                    ast.Str(s="bucket-for-microblend-compiler"),
                    ast.Name(id=each_input_param, ctx=ast.Load()),
                ],
                keywords=[],
            )
        )
        upload_obj_list.append(upload_obj)

    return upload_obj_list


def make_obj_for_downloading_from_s3(input_param_list):
    """
    s3_client.download("/tmp/")
    """
    download_obj_list = []

    for each_input_param in input_param_list:
        download_obj = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="s3_client", ctx=ast.Load()),
                    attr="download_file",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Str(s="input-output-microblend-bucket"),
                    ast.Name(id=each_input_param, ctx=ast.Load()),
                    ast.BinOp(
                        left=ast.Str(s="/tmp/"),
                        op=ast.Add(),
                        right=ast.Name(id=each_input_param, ctx=ast.Load()),
                    ),
                ],
                keywords=[],
            )
        )
        download_obj_list.append(download_obj)

    return download_obj_list


def add_using_s3_in_lambda_handler(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Add downloading input from s3 in lambda handler depending on input.
    e.g., put s3_client.download()
    """

    logger.info("Add using s3 and s3 client in lambda handler")

    if whole_ast_info.offloading_whole_application:
        process_offloading_whole_app_lambda(whole_ast_info)
        return

    process_each_compiler_generated_lambda(whole_ast_info)


def process_each_compiler_generated_lambda(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Process each compiler-generated lambda based on its type.
    """

    func_name_to_function_info_class_dict = whole_ast_info.function_information
    lambda_function_info = whole_ast_info.lambda_function_info

    for _, each_compiler_generated_lambda in lambda_function_info.items():
        if isinstance(each_compiler_generated_lambda, CompilerGeneratedLambda):
            process_compiler_generated_lambda(each_compiler_generated_lambda, func_name_to_function_info_class_dict)
        elif isinstance(each_compiler_generated_lambda, MergedCompilerGeneratedLambda):
            process_merged_compiler_generated_lambda(each_compiler_generated_lambda)


def process_offloading_whole_app_lambda(whole_ast_info):
    """
    Process offloading whole app lambda
    """
    lambda_module_for_whole = whole_ast_info.module_info_for_offloading_whole_app
    lamb_func_object = lambda_module_for_whole.lambda_handler_func_object
    lambda_module = lambda_module_for_whole.lambda_module

    add_event_input_assign_objs_in_lambda_handler(lamb_func_object, lambda_module_for_whole)
    add_upload_to_s3_for_return_object(lamb_func_object, lambda_module_for_whole)

    logger.info("Add s3_client and lambda_client if needed")

    if lambda_module_for_whole.lambda_event_input_objs:
        logger.info("Add s3_client")
        import_name_list = list(whole_ast_info.import_information.keys())
        logger.debug(import_name_list)

        if "boto3" not in import_name_list:
            handle_boto3_import(
                lambda_module,
                whole_ast_info,
                import_name_list,
                lambda_module_for_whole,
            )


def handle_boto3_import(lambda_module, whole_ast_info, import_name_list, lambda_module_for_whole):
    """
    Handle boto3 import
    """
    logger.info("Add import boto3")
    add_import_boto3_in_lambda_module(lambda_module, whole_ast_info)
    logger.info("Add S3 client")
    add_s3_client_in_lambda_module(lambda_module, whole_ast_info)
    import_name_list.append("boto3")
    lambda_module_for_whole.imports = import_name_list


def process_compiler_generated_lambda(each_compiler_generated_lambda, func_name_to_function_info_class_dict):
    """
    Process compiler generated lambda.
    """
    original_func_name = each_compiler_generated_lambda.original_func_name
    original_func_info = func_name_to_function_info_class_dict.get(original_func_name)
    lambda_func_object = each_compiler_generated_lambda.lambda_handler_func_object

    insert_objects_for_downloading_input(lambda_func_object,
                                         original_func_info.parameters,
                                         each_compiler_generated_lambda.lambda_event_input_objs)

    if return_objects := original_func_info.return_values:
        put_return_objs_for_s3(lambda_func_object, return_objects)


def process_merged_compiler_generated_lambda(each_compiler_generated_lambda):
    """
    Process merged compiler generated lambda.
    """
    lambda_func_object = each_compiler_generated_lambda.lambda_handler_func_object

    for each_ast in lambda_func_object.body:
        if isinstance(each_ast, ast.If):
            input_arguments = each_compiler_generated_lambda.lambda_input_per_if_statement.get(each_ast)
            event_input_obj_list = each_compiler_generated_lambda.parse_input_per_if_statement.get(each_ast)

            insert_objects_for_downloading_input(each_ast,
                                                 input_arguments,
                                                 event_input_obj_list)

            if return_objects := each_compiler_generated_lambda.return_obj_per_if_statement.get(each_ast):
                put_return_objs_for_s3(each_ast, return_objects)


def insert_objects_for_downloading_input(lambda_func_object, input_parameters, event_input_objs):
    """
    Inserts objects for downloading from S3 into the lambda function object.
    """
    object_for_downloading_input = make_obj_for_downloading_from_s3(input_parameters)
    last_event_input_obj_index = lambda_func_object.body.index(event_input_objs[-1]) + 1

    for idx, each_object_for_downloading_input in enumerate(object_for_downloading_input):
        lambda_func_object.body.insert(idx + last_event_input_obj_index, each_object_for_downloading_input)


def add_upload_to_s3_for_return_object(lambda_func_object, lambda_module_for_whole):
    """
    Add upload to s3 for return object
    """
    if lambda_module_for_whole.return_values:
        logger.info("Handling return objects")
        put_return_objs_for_s3(lambda_func_object, lambda_module_for_whole.return_values)


def add_s3_client_in_lambda_module(lambda_module, whole_ast_info):
    """
    Add s3 client in lambda module
    """
    non_func_info = whole_ast_info.non_function_object_info
    sorted_non_func_info = sorted(
        non_func_info.items(), key=lambda k_v: k_v[1].line_number
    )
    last_non_func_info = sorted_non_func_info[-1][1]
    last_non_func_obj = last_non_func_info.ast_object
    index_for_last_non_func_obj = lambda_module.body.index(last_non_func_obj)
    s3_client_object = make_s3_client()
    lambda_module.body.insert(
        index_for_last_non_func_obj + 1, s3_client_object,
    )


def add_import_boto3_in_lambda_module(lambda_module, whole_ast_info):
    """
    Add import boto3 in lambda module
    """
    # Fetch import information
    import_info = whole_ast_info.import_information
    # Sort by line number
    sorted_import_info = sorted(import_info.items(), key=lambda k_v: k_v[1].line_number)
    # Find the index for last import ast object
    last_import_obj_index = find_last_import_obj_index(lambda_module.body, sorted_import_info)
    # Put import boto3
    lambda_module.body.insert(
        last_import_obj_index + 1,
        whole_ast_info.boto3_and_json_imports.get("boto3").ast_object,
    )


def find_last_import_obj_index(module_body, sorted_import_info):
    """
    Find the index for last import ast object
    """
    for index, ast_node in enumerate(module_body):
        if isinstance(ast_node, ast.ImportFrom):
            import_info = ast_node.module, ast_node.level, ast_node.names
            if import_info in sorted_import_info:
                return index
    return -1


def add_event_input_assign_objs_in_lambda_handler(lambda_func_object, lambda_module_for_whole):
    """
    Add event input assign objects in lambda handler
    """
    lambda_event_input_objs = lambda_module_for_whole.lambda_event_input_objs
    last_event_input_obj_index = find_last_event_input_obj_index(lambda_func_object.body, lambda_event_input_objs)
    object_for_downloading_input = make_obj_for_downloading_from_s3(lambda_module_for_whole.input_parameter)
    insert_objects_after_index(lambda_func_object.body, last_event_input_obj_index, object_for_downloading_input)


def find_last_event_input_obj_index(body, lambda_event_input_objs):
    """
    Find last event input obj index
    """
    return next(
        (
            index
            for index, ast_node in enumerate(body)
            if ast_node in lambda_event_input_objs
        ),
        -1,
    )


def insert_objects_after_index(body, index, objects_to_insert):
    """
    Insert objects after index
    """
    for idx, obj in enumerate(objects_to_insert):
        body.insert(index + idx + 1, obj)


def put_return_objs_for_s3(lambda_func_object, return_objects):
    """
    Put return objects for s3
    """
    obj_for_uploading_to_s3 = make_obj_for_uploading_return_obj_to_s3(return_objects)
    for each_obj in obj_for_uploading_to_s3:
        lambda_func_object.body.insert(len(lambda_func_object.body) - 1, each_obj)


def make_module_for_lambda_handler(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Make module for each lambda handler, add s3 client
    """

    logger.info("Make ast.Module for each lambda function")

    if not whole_ast_info.offloading_whole_application:
        make_lambda_handler_modules(whole_ast_info)


def make_lambda_handler_modules(whole_ast_info):
    """
    Make lambda handler modules by setting up necessary imports and merging different parts of the code
    to form the complete lambda function.
    """

    # Fetch information from the whole_ast_info object
    import_info_dict_name_to_class = whole_ast_info.import_information
    non_func_obj_info = whole_ast_info.non_function_object_info
    func_name_to_function_info_class_dict = whole_ast_info.function_information
    lambda_handlers_info = whole_ast_info.lambda_function_info

    # Extract import information and AST objects from non-function objects
    non_func_import_name_list, non_func_ast_list = set(), []
    for non_func_ast, non_func_info in non_func_obj_info.items():
        non_func_import_name_list.update(non_func_info.imports)
        non_func_ast_list.append(non_func_ast)

    # Create an S3 client object, likely for use within the lambda
    s3_client_object = make_s3_client()

    # Iterate through all lambda handlers and set up necessary imports, modules, and other components
    for l_group_name, each_lambda_handler in lambda_handlers_info.items():
        if l_group_name != "Default":
            import_objects_list = []
            import_name_list = []
            import_info_dict = defaultdict()

            # Accumulate all the imports from the function definitions that are part of the lambda
            function_name_list = each_lambda_handler.lambda_name_list
            for each_function in function_name_list:
                original_func_name_info = func_name_to_function_info_class_dict.get(each_function)
                import_name_list.extend(original_func_name_info.imports)
                import_name_list.extend(non_func_import_name_list)

            # Gather the AST objects for all the imports
            for each_import_name in import_name_list:
                if each_import_name != "boto3":
                    import_info = import_info_dict_name_to_class.get(each_import_name)
                    import_ast_object = import_info.ast_object
                    import_info_dict[each_import_name] = import_ast_object
                    import_objects_list.append(import_ast_object)

            # If boto3 is not already in the list, add it
            if "boto3" not in import_name_list:
                boto3_info = whole_ast_info.boto3_and_json_imports["boto3"]
                import_objects_list.append(boto3_info.ast_object)
                non_func_ast_list.append(s3_client_object)

            # Removing duplicate imports and AST objects
            new_import_objects_list = list(set(import_objects_list))
            new_non_func_ast_list = list(set(non_func_ast_list))

            # Update the lambda handler object with the new imports and structure
            each_lambda_handler.import_info_dict = import_info_dict
            each_lambda_handler.lambda_module.body = (
                    new_import_objects_list + new_non_func_ast_list + each_lambda_handler.lambda_module.body
            )

            # Logging for debugging purposes
            logger.info(astor.to_source(each_lambda_handler.lambda_module))
            logger.debug(each_lambda_handler.lambda_group_name)

            # Update imports and add the completed module to the whole AST info
            each_lambda_handler.imports = new_import_objects_list
            whole_ast_info.lambda_handler_module_dict[
                each_lambda_handler.lambda_group_name] = each_lambda_handler.lambda_module


def make_s3_client():
    """
    Make s3 client
    """
    boto3_attribute_object = ast.Attribute(
        value=ast.Name(id="boto3", ctx=ast.Load()), attr="client", ctx=ast.Load()
    )
    s3_client_name_object = ast.Name(id="s3_client", ctx=ast.Store())
    boto3_call_object = ast.Call(
        func=boto3_attribute_object,
        args=[ast.Str(s="s3")],
        keywords=[
            ast.keyword(arg="region_name", value=ast.Str(s="us-east-1")),
            ast.keyword(arg=None, value=ast.Name(id="CREDENTIALS", ctx=ast.Load())),
        ],
    )
    s3_client_object = ast.Assign(
        targets=[s3_client_name_object], value=boto3_call_object
    )
    s3_client_object = ast.fix_missing_locations(s3_client_object)
    return s3_client_object


def make_lambda_code_directory(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Create a directory structure for lambda functions. Each directory will have its respective lambda handler.
    """
    lambda_code_path = "./lambda_path"
    logger.info("Make directory for lambda functions")

    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app] : Make only one directory")
        return

    _initialize_lambda_directory(lambda_code_path)

    lambda_info_dict = whole_ast_info.lambda_function_info
    _handle_lambda_information(lambda_code_path, lambda_info_dict)

    if lambda_handler_module_dict := whole_ast_info.lambda_handler_module_dict:
        _handle_lambda_handler_modules(lambda_code_path, lambda_handler_module_dict)
    else:
        return


def _initialize_lambda_directory(lambda_code_path: str) -> None:
    """
    Initialize a directory for lambda codes. If it exists, clean it. Otherwise, create it.
    """
    if os.path.exists(lambda_code_path):
        logger.info(f"Empty {lambda_code_path}")
        shutil.rmtree(lambda_code_path)
    os.makedirs(lambda_code_path)


def _handle_lambda_information(lambda_code_path: str, lambda_info_dict: Dict[str, Any]) -> None:
    """
    Handle the generation of directories and files for each lambda information.
    """
    for lambda_name, lambda_info in lambda_info_dict.items():
        if isinstance(lambda_info, MergedCompilerGeneratedLambda):
            lambda_module_obj = lambda_info.lambda_module
            _generate_lambda_directory_and_file(lambda_code_path, lambda_name, lambda_module_obj)


def _handle_lambda_handler_modules(lambda_code_path: str, lambda_handler_module_dict: Dict[str, ast.Module]) -> None:
    """
    Handle the generation of directories and files for each lambda handler module.
    """
    for lambda_handler_name, lambda_handler_module in lambda_handler_module_dict.items():
        _generate_lambda_directory_and_file(lambda_code_path, lambda_handler_name, lambda_handler_module)


def _generate_lambda_directory_and_file(lambda_code_path: str, lambda_name: str, module_obj: Any) -> None:
    """
    Generate a directory and a file for a given lambda name and module object.
    """
    sub_dir = os.path.join(lambda_code_path, lambda_name)
    os.makedirs(sub_dir)

    native_code = astor.to_source(module_obj)

    with open(os.path.join(sub_dir, f"{lambda_name}.py"), "w") as temp_file:
        temp_file.write(native_code)


def insert_imports_in_lambda_code_folder(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Insert module(import) for each lambda handler in each directory.
    This function ensures that the necessary import modules are present in the lambda code directory.
    """

    logger.info("Link library modules for lambda handlers")

    if whole_ast_info.offloading_whole_application:
        offload_whole_application(whole_ast_info)
        return

    # Fetch the list of import modules present in the directory
    import_modules_dir, module_lists = bring_module_list_in_import_modules_folder()

    # Define the base path where lambda codes are located
    lambda_code_path = "./lambda_path"
    lambda_info_dict = whole_ast_info.lambda_function_info

    # Iterate through each lambda handler and add necessary import modules
    for lambda_name, lambda_info in lambda_info_dict.items():
        if not isinstance(lambda_info, MergedCompilerGeneratedLambda):
            continue

        # Identify the directory specific to the lambda handler
        lambda_dir = os.path.join(lambda_code_path, lambda_info.lambda_group_name)

        # Retrieve the list of imports used within the lambda handler
        import_list = lambda_info.import_name_list

        # Handle library dependencies, if any (e.g., mxnet, matplotlib)
        consider_dependency_between_modules(import_list)

        # Group the imports based on whether they are Python scripts or not
        import_group_by_python_script = group_import_by_from_script(whole_ast_info)

        # Log the grouped imports
        module_list = list(import_group_by_python_script.values())[0]
        logger.info(f"Import modules : {module_list}")

        # For each import, ensure it's present within the lambda directory
        iterate_and_put_import_modules(
            import_group_by_python_script,
            import_modules_dir,
            lambda_dir,
            module_lists
        )


def offload_whole_application(whole_ast_info):
    """
    Offload whole application
    """
    logger.info("[Offloading whole app] : Link library modules")

    # Fetch import modules in directory
    import_modules_dir, module_lists = bring_module_list_in_import_modules_folder()

    # Fetch module
    module_to_offload = whole_ast_info.module_info_for_offloading_whole_app

    # Combine lambda dir with lambda name
    lambda_dir = os.path.join("./lambda_path", module_to_offload.lambda_name)

    # Fetch all import modules used in this source code
    import_list = module_to_offload.imports

    # In case of library dependency e.g., mxnet, matplotlib
    consider_dependency_between_modules(import_list)

    # Group import name depending on python script or not
    import_group_by_python_script = group_import_by_from_script(whole_ast_info)

    module_list = list(import_group_by_python_script.values())[0]
    logger.info(f"Import modules: {module_list}")

    # Iterate each import name in source code
    iterate_and_put_import_modules(
        import_group_by_python_script, import_modules_dir, lambda_dir, module_lists
    )


def iterate_and_put_import_modules(import_group_by_python_script, import_modules_dir, lambda_dir, module_lists):
    """
    Iterate each import name in source code and put import modules in lambda directory
    """
    for from_python_script, import_list in import_group_by_python_script.items():
        for each_import_name in import_list:
            if from_python_script:  # If from python script
                python_script_dir = os.path.join(os.getcwd(), f"{each_import_name}.py")
                shutil.copy(python_script_dir, lambda_dir)
            else:  # Default import modules
                for each_module in module_lists:
                    if each_module.lower().startswith(each_import_name.lower()):
                        module_dir = os.path.join(import_modules_dir, each_module)
                        shutil.copytree(module_dir, lambda_dir)
                        break


def group_import_by_from_script(whole_ast_info):
    """
    Group import name depending on from_python_script

    """
    import_info = list(whole_ast_info.import_information.values())
    lambda_dict_for_python_script = defaultdict(list)
    for x in import_info:
        lambda_dict_for_python_script[x.from_python_script].append(x.import_name)
    return dict(lambda_dict_for_python_script)


def consider_dependency_between_modules(import_list) -> None:
    """
    Adjusts the import_list based on module dependencies.

    Args:
        import_list (set): A set of module names.
    """

    # Remove dependencies if 'matplotlib' is in the list
    if "matplotlib" in import_list:
        import_list.discard("matplotlib")
        import_list.discard("numpy")

    # Add dependencies if 'mxnet' is in the list
    if "mxnet" in import_list:
        import_list.update(["requests", "urllib3", "chardet"])


def bring_module_list_in_import_modules_folder():
    """
    Bring module list in import_modules folder
    """
    import_modules_dir = "../Workload/SocialNetwork"
    module_lists = next(os.walk(os.path.join(os.getcwd(), import_modules_dir)))[1]
    return import_modules_dir, module_lists


def make_lambda_function(lambda_deploy_info, whole_info):
    """
    Use boto3 lambda_client to create or remove functions
    """

    lambda_client = boto3.client("lambda", **CREDENTIALS)
    lambda_name_to_make = lambda_deploy_info.lambda_name_to_make_on_aws

    try:
        lambda_client.get_function(FunctionName=lambda_name_to_make)

    except lambda_client.exceptions.ResourceNotFoundException:
        logger.info("Creating lambda function")
        create_lambda_function_by_cli(lambda_deploy_info, whole_info)
        logger.info(f"Created {lambda_deploy_info.lambda_name_to_make_on_aws}")

    else:
        logger.info("Function already exists, removing current function")
        lambda_client.delete_function(FunctionName=lambda_name_to_make)
        logger.info(f"Deleted lambda function named - {lambda_name_to_make}")
        create_lambda_function_by_cli(lambda_deploy_info, whole_info)
        logger.info(f"Created lambda function named - {lambda_name_to_make}")


def create_lambda_function_by_cli(lambda_deploy_info, whole_info):
    lambda_client = boto3.client("lambda", region_name="us-east-1", **CREDENTIALS)

    function_name = lambda_deploy_info.lambda_name_to_make_on_aws
    s3_bucket = "./compiler_module_config.bucket_for_lambda_handler_zip"
    handler_name = lambda_deploy_info.handler_name

    if whole_info.offloading_whole_application:
        object_in_s3 = whole_info.deployment_name_for_offloading_whole_app
        logger.info(f"using zip named - {object_in_s3}")
        lambda_client.create_function(
            Code={
                "S3Bucket": s3_bucket,
                "S3Key": object_in_s3,
            },
            FunctionName=function_name,
            Handler=f"{handler_name}_{function_name}.{handler_name}_{function_name}",
            MemorySize=lambda_deploy_info.memory,
            Publish=True,
            Role=lambda_deploy_info.aws_role,
            Runtime=lambda_deploy_info.runtime_version,
            Timeout=lambda_deploy_info.time_out,
        )

    else:
        s3_key = lambda_deploy_info.zip_file_name_in_s3
        lambda_client.create_function(
            Code={
                "S3Bucket": s3_bucket,
                "S3Key": s3_key,
            },
            FunctionName=function_name,
            Handler=f"{handler_name}.{handler_name}",
            MemorySize=lambda_deploy_info.memory,
            Publish=True,
            Role=lambda_deploy_info.aws_role,
            Runtime=lambda_deploy_info.runtime_version,
            Timeout=lambda_deploy_info.time_out,
        )


def map_func_to_func_arn(whole_ast_info: WholeASTInfoClass) -> None:
    logger.info("Make assignment object that maps function names to function name ARNs")

    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app] : Skip since no function call orchestrator")
    else:
        lambda_deployment_zip_info = whole_ast_info.lambda_deployment_zip_info

        if not lambda_deployment_zip_info:
            return

        logger.debug(lambda_deployment_zip_info)

        annotated_code_module = whole_ast_info.copied_module_for_analysis

        original_func_list = []
        lambda_func_name_in_aws = []

        for _, deployment_info in lambda_deployment_zip_info.items():
            original_func_list.append(ast.Str(s=deployment_info.original_func_name))
            logger.debug(deployment_info.original_func_name)
            lambda_func_name_in_aws.append(ast.Str(s=deployment_info.lambda_name_to_make_on_aws))
            logger.debug(deployment_info.lambda_name_to_make_on_aws)

        last_non_func_obj_info: NonFunctionInfoClass = list(whole_ast_info.non_function_object_info.values())[-1]
        last_non_func_obj = last_non_func_obj_info.ast_object
        index_for_last_non_func_object = annotated_code_module.body.index(last_non_func_obj) + 1

        map_func_to_func_arn_object = ast.Assign(
            targets=[ast.Name(id="map_func_to_func_arn_dict", ctx=ast.Store())],
            value=ast.Dict(keys=original_func_list, values=lambda_func_name_in_aws),
            lineno=last_non_func_obj.lineno + 1,
            col_offset=0,
        )
        logger.debug(astor.to_source(map_func_to_func_arn_object))

        whole_ast_info.map_func_to_func_arn_object = map_func_to_func_arn_object
        logger.debug(astor.to_source(annotated_code_module))
    return


def write_hybrid_code(whole_ast_info: WholeASTInfoClass):
    """
    Make a code that is hybrid-case
    1. Make hybrid code using copy
    2. Put Function of Invoke Call
    3. Change Function Call to Function Call of Invoke Call
    4. Add boto3, json import and (lambda and s3) client
    """

    logger.info("Making VM Hybrid")

    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app] : Skip Making VM Hybrid")
        return

    annotated_code_module: ast.Module = whole_ast_info.copied_module_for_analysis
    non_func_info = whole_ast_info.non_function_object_info
    invoke_func_object = whole_ast_info.lambda_invoke_function
    lambda_based_func_call_info_list = whole_ast_info.func_call_using_lambda
    combined_func_call_using_lambda = whole_ast_info.combined_func_call_using_lambda

    # Make copy of annotated code since copied_module_for_analysis will be  modified
    annotated_code_module_for_user = copy.deepcopy(annotated_code_module)
    whole_ast_info.annotated_code_module_for_user = annotated_code_module_for_user

    # Fetch map_func_to_func_arn_object
    map_func_to_func_arn_object = whole_ast_info.map_func_to_func_arn_object

    # Find the index of last non-fun and put invoke function after that index
    last_non_func_object_info: NonFunctionInfoClass = list(non_func_info.values())[-1]
    last_non_func_obj = last_non_func_object_info.ast_object
    last_non_func_obj_index = annotated_code_module.body.index(last_non_func_obj) + 1
    if map_func_to_func_arn_object:
        annotated_code_module.body = (
                annotated_code_module.body[:last_non_func_obj_index]
                + [map_func_to_func_arn_object]
                + [invoke_func_object]
                + annotated_code_module.body[last_non_func_obj_index:]
        )

    else:
        annotated_code_module.body = (
                annotated_code_module.body[:last_non_func_obj_index]
                + [invoke_func_object]
                + annotated_code_module.body[last_non_func_obj_index:]
        )

    # Change function call to function call of invoke call
    each_func_call: LambdaBasedFunctionCallInfoClass

    # Fetch function call that uses lambda-based function call
    for each_func_call in lambda_based_func_call_info_list:
        # Bring original and lambda_based func call  -> replace them
        origin_func_call: FunctionCallInfo = each_func_call.original_func_call_info
        copied_func_call = each_func_call.copied_func_call_info
        origin_func_call_object = origin_func_call.ast_object
        copied_func_call_object = copied_func_call.ast_object

        # Use transformer to replace assign or expr object with newly made
        change_func_call = ChangeFunctionCallTransformer(
            origin_func_call_object, copied_func_call_object
        )
        annotated_code_module = change_func_call.visit(annotated_code_module)
        annotated_code_module = ast.fix_missing_locations(annotated_code_module)

    for _, function_call_list in combined_func_call_using_lambda.items():

        first_func_call = function_call_list[0]
        origin_func_call: FunctionCallInfo = first_func_call.original_func_call_info
        copied_func_call = first_func_call.copied_func_call_info
        origin_func_call_object = origin_func_call.ast_object
        copied_func_call_object = copied_func_call.ast_object

        change_func_call = ChangeFunctionCallTransformer(
            origin_func_call_object, copied_func_call_object
        )
        annotated_code_module = change_func_call.visit(annotated_code_module)
        annotated_code_module = ast.fix_missing_locations(annotated_code_module)

        remaining_func_call = function_call_list[1:]
        for each_func_call in remaining_func_call:
            origin_func_call: FunctionCallInfo = each_func_call.original_func_call_info
            # copied_func_call = last_func_call.copied_func_call_info
            origin_func_call_object = origin_func_call.ast_object
            # copied_func_call_object = copied_func_call.ast_object

            change_func_call = RemoveNodeTransformer(origin_func_call_object)
            annotated_code_module = change_func_call.visit(annotated_code_module)
            annotated_code_module = ast.fix_missing_locations(annotated_code_module)

        import_info_dict = whole_ast_info.import_information
        import_name_list_in_hybrid = list(import_info_dict.keys())
        last_import_info_in_hybrid = list(import_info_dict.values())[-1].ast_object

        import_name_to_add = [
            x for x in ["boto3", "json"] if x not in import_name_list_in_hybrid
        ]
        for each_import in import_name_to_add:
            module_to_add = ast.fix_missing_locations(
                ast.Import(names=[ast.alias(name=each_import, asname=None)])
            )
            last_import_index = (
                    annotated_code_module.body.index(last_import_info_in_hybrid) + 1
            )
            annotated_code_module.body = (
                    annotated_code_module.body[:last_import_index]
                    + [module_to_add]
                    + annotated_code_module.body[last_import_index:]
            )
            annotated_code_module = ast.fix_missing_locations(annotated_code_module)

        lambda_client_object = make_lambda_client()
        s3_client_object = make_s3_client()

        last_non_func_obj_index = (
                annotated_code_module.body.index(map_func_to_func_arn_object) + 1
        )
        # Append lambda client to last non function object
        annotated_code_module.body = (
                annotated_code_module.body[:last_non_func_obj_index]
                + [lambda_client_object]
                + [s3_client_object]
                + annotated_code_module.body[last_non_func_obj_index:]
        )

    # Add lambda or s3 client with import if doesn't exist
    if lambda_based_func_call_info_list:

        # Fetch import list information, import_name_list, last_import_object
        import_info_dict = whole_ast_info.import_information
        import_name_list_in_hybrid = list(import_info_dict.keys())
        last_import_info_in_hybrid = list(import_info_dict.values())[-1].ast_object

        # json or boto3 when hybrid code doesn't have one
        import_name_to_add = [
            x for x in ["boto3", "json"] if x not in import_name_list_in_hybrid
        ]

        # Add import boto3 or json
        for each_import in import_name_to_add:
            module_to_add = ast.fix_missing_locations(
                ast.Import(names=[ast.alias(name=each_import, asname=None)])
            )
            last_import_index = (
                    annotated_code_module.body.index(last_import_info_in_hybrid) + 1
            )
            annotated_code_module.body = (
                    annotated_code_module.body[:last_import_index]
                    + [module_to_add]
                    + annotated_code_module.body[last_import_index:]
            )
            annotated_code_module = ast.fix_missing_locations(annotated_code_module)

        # Add lambda and s3 client using boto3
        lambda_client_object = make_lambda_client()
        s3_client_object = make_s3_client()

        # Append lambda client to last non function object
        annotated_code_module.body = (
                annotated_code_module.body[:last_non_func_obj_index]
                + [lambda_client_object]
                + [s3_client_object]
                + annotated_code_module.body[last_non_func_obj_index:]
        )

        # Add lambda client object to non_func_info class
        non_func_info[lambda_client_object] = NonFunctionInfoClass(
            last_non_func_obj.lineno + 1,
            ast_object=lambda_client_object,
            object_type="Assign",
            description="lambda_client object",
        )

        non_func_info[s3_client_object] = NonFunctionInfoClass(
            last_non_func_obj.lineno + 1,
            ast_object=s3_client_object,
            object_type="Assign",
            description="s3_client_object",
        )


def make_lambda_client():
    """
    Make lambda client object
    """
    boto3_attribute_object = ast.Attribute(
        value=ast.Name(id="boto3", ctx=ast.Load()), attr="client", ctx=ast.Load()
    )
    lambda_client_name_object = ast.Name(id="lambda_client", ctx=ast.Store())
    boto3_call_object = ast.Call(
        func=boto3_attribute_object,
        args=[ast.Str(s="lambda")],
        keywords=[
            ast.keyword(arg="region_name", value=ast.Str(s="us-east-1")),
            ast.keyword(arg=None, value=ast.Name(id="CREDENTIALS", ctx=ast.Load())),
        ],
    )
    lambda_client_object = ast.Assign(
        targets=[lambda_client_name_object], value=boto3_call_object
    )
    lambda_client_object = ast.fix_missing_locations(lambda_client_object)
    return lambda_client_object


class ChangeFunctionCallTransformer(ast.NodeTransformer):
    def __init__(self, original_ast, copied_ast):
        self.original_ast = original_ast
        self.copied_ast = copied_ast

    def visit_Assign(self, node):
        if node is self.original_ast:
            node = self.copied_ast

        self.generic_visit(node)
        return node

    def visit_Expr(self, node):
        if node is self.original_ast:
            node = self.copied_ast

        self.generic_visit(node)
        return node


class RemoveNodeTransformer(ast.NodeTransformer):
    def __init__(self, original_ast):
        self.original_ast = original_ast

    def visit_Assign(self, node):
        return None if node is self.original_ast else node

    def visit_Expr(self, node):
        return None if node is self.original_ast else node


def save_hybrid_code_in_output_directory(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Write copied_module_for_analysis which is hybrid code to hybrid_code directory
    """
    logger.info("Save hybrid code to hybrid_code directory")

    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app] : Skipping since offloading whole app")
        return

    hybrid_code_to_write = whole_ast_info.copied_module_for_analysis
    hybrid_code_dir = "./lambda_hybrid_code_path"
    hybrid_code_file_name = "./compiler_module_config.hybrid_code_file_name"

    # remove directory if already exists
    if os.path.exists(hybrid_code_dir):
        shutil.rmtree(hybrid_code_dir)
    os.makedirs(hybrid_code_dir)

    # write code to directory
    hybrid_code = astor.to_source(hybrid_code_to_write)
    with open(os.path.join(hybrid_code_dir, f"{hybrid_code_file_name}.py"), "w") as temp_file:
        temp_file.write(hybrid_code)


def empty_output():
    output_dir = "./compiler_module_config.output_path_dir"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


def make_zip_file_for_lambda_handler(whole_ast_info: WholeASTInfoClass) -> None:
    """
    Add directory of "deployment_zip_dir"
    """
    logger.info("Zip library modules and code for deployment")

    # Delete if exists and make directory
    deployment_zip_dir = "./compiler_module_config.deployment_zip_dir"
    if whole_ast_info.offloading_whole_application:
        make_zip_for_whole_app(whole_ast_info, deployment_zip_dir)
    else:
        make_zip_for_general_case(whole_ast_info, deployment_zip_dir)


def make_zip_for_general_case(whole_ast_info, deployment_zip_dir):
    """
    Make zip file for general case
    """
    lambda_handler_module_dict: Dict[str, ast.Module] = whole_ast_info.lambda_handler_module_dict

    lambda_code_dir_path = make_lambda_zip_path(deployment_zip_dir)
    logger.debug(lambda_code_dir_path)

    # Find lambda handler directories in lambda_code_dir_path
    lambda_dir_list = [
        dI
        for dI in os.listdir(lambda_code_dir_path)
        if os.path.isdir(os.path.join(lambda_code_dir_path, dI))
    ]
    logger.debug(os.getcwd())
    logger.debug(lambda_dir_list)

    # Zip each lambda handler directory and move it to deployment_zip_dir
    for each_lambda_dir in lambda_dir_list:
        shutil.make_archive(
            str(each_lambda_dir),
            "zip",
            os.path.join(lambda_code_dir_path, each_lambda_dir),
        )
        shutil.move(
            f"{str(each_lambda_dir)}.zip",
            os.path.join(os.getcwd(), deployment_zip_dir),
        )


def make_lambda_zip_path(deployment_zip_dir):
    """
    Make lambda zip path
    """
    if os.path.exists(deployment_zip_dir):
        os.system(f"rm -rf {deployment_zip_dir}")
    os.makedirs(deployment_zip_dir)
    return "./lambda_path"


def make_zip_for_whole_app(whole_ast_info: WholeASTInfoClass, deployment_zip_dir: str) -> None:
    """
    Make zip file for whole app
    """
    logger.info("[Offloading whole app] : Making only one zip")

    lambda_dir = make_lambda_zip_path(deployment_zip_dir)
    module_info = whole_ast_info.module_info_for_offloading_whole_app
    lambda_name = module_info.lambda_name

    # Zip code with modules
    zip_file_path = os.path.join(lambda_dir, f"{lambda_name}.zip")
    shutil.make_archive(lambda_name, "zip", os.path.join(lambda_dir, lambda_name))

    # Move zip file to deployment_zip_dir
    shutil.move(zip_file_path, os.path.join(os.getcwd(), deployment_zip_dir))
    logger.info("Zipped Module")


def make_lambda_function_using_aws_cli(whole_ast_info: WholeASTInfoClass):
    """
    use boto3 to make lambda for each compiler generated lambda handler
    """
    logger.info("Create lambda function using aws cli")

    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app] : Create only one lambda")

        os.chdir("./compiler_module_config.deployment_zip_dir")

        original_file_name = whole_ast_info.file_name
        original_file_name_without_extension = original_file_name.split(".")[0]

        logger.debug(original_file_name)

        try:
            lambda_name = whole_ast_info.lambda_config_for_whole_application.func_name
        except AttributeError:
            lambda_name = original_file_name_without_extension
            logger.info(f"Using file name to {lambda_name}")

        lambda_handler_name = "lambda_handler"
        lambda_zip_name = f"{lambda_handler_name}.zip"
        lambda_arn = f"arn:aws:lambda:us-east-1:xxxx:function:{lambda_name}"

        lambda_deploy_info = LambdaDeployInfo(
            original_func_name=original_file_name,
            lambda_zip_name=lambda_zip_name,
            lambda_name_to_make_on_aws=lambda_name,
            lambda_arn=lambda_arn,
            handler_name=lambda_handler_name,
            zip_file_name_in_s3=whole_ast_info.deployment_name_for_offloading_whole_app,
        )

        whole_ast_info.lambda_deployment_zip_info[lambda_handler_name] = lambda_deploy_info

        make_lambda_function(lambda_deploy_info, whole_ast_info)

        os.chdir("../../..")

        return

    else:
        lambda_info_dict = whole_ast_info.lambda_function_info

        if not lambda_info_dict:
            if os.path.exists("./compiler_module_config.deployment_zip_dir"):
                os.system("rm -rf ./compiler_module_config.deployment_zip_dir")
            return

        for lambda_name, lambda_info in lambda_info_dict.items():
            os.chdir("./compiler_module_config.deployment_zip_dir")

            if isinstance(lambda_info, MergedCompilerGeneratedLambda):
                zip_file_name_in_s3_bucket = whole_ast_info.lambda_deployment_zip_info[lambda_info.lambda_group_name]
                original_file_name = lambda_info.lambda_name_list
                lambda_name_for_creation = "_and_".join(original_file_name)
                lambda_zip_name = f"{lambda_info.lambda_group_name}.zip"
                lambda_name_to_make = f"microblend_{lambda_name_for_creation}"
                lambda_arn = f"arn:aws:lambda:us-east-1:206135129663:function:{lambda_name_to_make}"

                lambda_deploy_info = LambdaDeployInfo(
                    original_func_name=lambda_name_for_creation,
                    lambda_zip_name=lambda_zip_name,
                    lambda_name_to_make_on_aws=lambda_name_to_make,
                    lambda_arn=lambda_arn,
                    handler_name=lambda_info.lambda_group_name,
                    zip_file_name_in_s3=zip_file_name_in_s3_bucket,
                )
                whole_ast_info.lambda_deployment_zip_info[lambda_info.lambda_group_name] = lambda_deploy_info

                os.chdir("../../..")

                return

        lambda_handler_module_dict: Dict[str, ast.Module] = whole_ast_info.lambda_handler_module_dict

        if lambda_handler_module_dict:
            os.chdir("./compiler_module_config.deployment_zip_dir")

            for lambda_handler_func_name, lambda_handler_info in whole_ast_info.compiler_generated_lambda_handler_dict.items():
                zip_file_name_in_s3_bucket = whole_ast_info.lambda_deployment_zip_info[lambda_handler_func_name]
                original_file_name = lambda_handler_info.original_func_name
                lambda_name_to_make = f"microblend_{original_file_name}"
                lambda_zip_name = f"{lambda_handler_func_name}.zip"
                lambda_arn = f"arn:aws:lambda:us-east-1:206135129663:function:{lambda_name_to_make}"
                handler_name_in_lambda_console = f"{lambda_handler_func_name}.{lambda_handler_func_name}"

                lambda_deploy_info = LambdaDeployInfo(
                    original_func_name=original_file_name,
                    lambda_zip_name=lambda_zip_name,
                    lambda_name_to_make_on_aws=lambda_name_to_make,
                    lambda_arn=lambda_arn,
                    handler_name=handler_name_in_lambda_console,
                    zip_file_name_in_s3=zip_file_name_in_s3_bucket,
                )

                whole_ast_info.lambda_deployment_zip_info[lambda_handler_func_name] = lambda_deploy_info

            os.chdir("../../..")


def upload_upload_hybrid_code_to_s3hybrid_code_to_s3(whole_ast_info: WholeASTInfoClass):
    """
    Upload hybrid code to S3
    """
    logger.info("Uploading hybrid code to s3")
    logger.info("\tSkipping since we aren't in dynamic phase")
    return


def upload_hybrid_code_to_s3(whole_ast_info: WholeASTInfoClass):
    """
    Upload hybrid code to S3, using aws cli
    """
    logger.info("Uploading hybrid code to s3")
    # upload the hybrid code to s3

    return


def upload_lambda_deployment_to_s3(whole_ast_info: WholeASTInfoClass):
    """
    Upload lambda deployment zip file to S3
    """
    logger.info("Uploading deployment to S3")
    if whole_ast_info.offloading_whole_application:
        logger.info("[Offloading whole app] : Uploading one zip file")

    # Fetch deployment directory
    deployment_dir = "./compiler_module_config.deployment_zip_dir"
    bucket_name = "./compiler_module_config.bucket_for_lambda_handler_zip"

    deploy_dir_list = os.listdir(deployment_dir)

    for each_deploy_zip in deploy_dir_list:
        upload_deploy_zip_to_s3(bucket_name, deployment_dir, each_deploy_zip, whole_ast_info)


def upload_deploy_zip_to_s3(bucket_name, deploy_directory, deploy_zip, whole_ast_info):
    """Randomize file name for consistency"""

    random_named_lambda_deploy_zip = f"{uuid.uuid4()}-{deploy_zip}"

    logger.info(f"Uploaded {random_named_lambda_deploy_zip}")

    # Save zip file name
    if whole_ast_info.offloading_whole_application:
        whole_ast_info.deployment_name_for_offloading_whole_app = random_named_lambda_deploy_zip
    else:
        whole_ast_info.lambda_deployment_zip_info[deploy_zip.split(".")[0]] = random_named_lambda_deploy_zip

    # TODO: The actual S3 uploading logic


def group_functions_by_lambda_group(ast_info: WholeASTInfoClass):
    """
    Group function information based on lambda group name
    """
    logger.info("Grouping functions by lambda group name.")

    # If offloading the whole application, there's no need to proceed with grouping.
    if ast_info.offloading_whole_application:
        logger.info("\tSkipping: offloading the entire application.")
        return

    # Extract function information
    functions = ast_info.function_information.values()

    # Group functions by lambda group name
    grouped_by_lambda = defaultdict(list)
    for function in functions:
        if function.func_name != "main" and function.initial_pragma == "FaaS":
            grouped_by_lambda[function.lambda_group_name].append(function.func_name)

    # Update the ast_info object with the grouped functions
    ast_info.sort_by_lambda_group = dict(grouped_by_lambda)

    # Logging grouped information
    if grouped_by_lambda:
        logger.info(f"\tGrouped by lambda: {grouped_by_lambda}")
    else:
        logger.info("\tNo lambda groups found.")


def extract_and_print_function(original_code, f_name):
    """
    Process the compiler on the original code
    :param original_code: The original code to be processed
    :param f_name: The name used for creating dataclass
    """

    # Create dataclass to store whole info
    ast_info = generate_original_and_copied_ast_info(original_code, f_name)

    # Copy module for analysis
    analyzed_ast_module: ast.Module = ast_info.copied_module_for_analysis

    # Find user annotations in the code
    extract_user_annotations(ast_info)

    function_candidates = []
    for each_func_info in ast_info.function_information.values():
        function_candidates.append(each_func_info.func_name)
    return function_candidates


def process_compiler(original_code, f_name):
    """
    Process the compiler on the original code
    :param original_code: The original code to be processed
    :param f_name: The name used for creating dataclass
    """

    # Create dataclass to store whole info
    ast_info = generate_original_and_copied_ast_info(original_code, f_name)

    # Copy module for analysis
    analyzed_ast_module: ast.Module = ast_info.copied_module_for_analysis

    # Find user annotations in the code
    extract_user_annotations(ast_info)

    logger.info("Function Information")
    for each_func_info in ast_info.function_information.values():
        logger.info(each_func_info.func_name)

    # Sort and group functions by lambda group name
    group_functions_by_lambda_group(ast_info)

    # Analyze imports and functions
    compiler_module_config = ModuleConfigClass(f_name)
    code_analyzer = ImportAndFunctionAnalyzer(ast_info, compiler_module_config)
    code_analyzer.visit(analyzed_ast_module)
    code_analyzer.start_analyzing()

    # Change function definitions to lambda functions
    transform_to_lambda_functions(ast_info)

    # Function call orchestrator and change function call
    function_call_orchestrator(ast_info)

    # Add modules for s3 in lambda handler
    add_using_s3_in_lambda_handler(ast_info)

    # Add modules for lambda handler
    make_module_for_lambda_handler(ast_info)

    # Make directory for each lambda handler
    make_lambda_code_directory(ast_info)

    # Put library modules in each lambda handler
    insert_imports_in_lambda_code_folder(ast_info)

    # Zip library modules and code for deployment
    make_zip_file_for_lambda_handler(ast_info)

    # Upload deployment list to s3
    upload_lambda_deployment_to_s3(ast_info)

    # Create lambda function using aws cli
    make_lambda_function_using_aws_cli(ast_info)

    # Make object that contains mapping func_name to lambda_func_arn
    map_func_to_func_arn(ast_info)

    # Making VM Hybrid
    write_hybrid_code(ast_info)

    # Write hybrid code to output directory
    save_hybrid_code_in_output_directory(ast_info)

    # Upload Hybrid to S3
    upload_hybrid_code_to_s3(ast_info)

    logger.info("End of Compiler")


def main():
    """
    Main function for compiler
    """
    source_file_name: str = "../Workload/SocialNetwork/app/media_service.py"
    logger.info(f"File is {source_file_name}")

    # open source file name
    with open(source_file_name, "r") as original_code:
        process_compiler(original_code, source_file_name)


if __name__ == "__main__":
    main()

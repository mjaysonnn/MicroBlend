import configparser
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pydantic

sys.path.append('utils')


def fetch_conf_ini():
    """
    Fetch config.ini
    """

    global conf_dict

    conf_dict = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.absolute() / 'utils' / 'config.ini'

    conf_dict.read(config_path)
    conf_dict = {sect: dict(conf_dict.items(sect)) for sect in conf_dict.sections()}
    conf_dict.pop('root', None)


conf_dict = {}

fetch_conf_ini()


@dataclass
class MicroserviceInfo:
    service_name: str
    service_loadbalancer_id: str
    service_loadbalancer_addr: str
    service_worker_filename: str
    service_port: int
    lambda_arn: str


@dataclass
class SocialNetworkInfo:
    compose_post_service_info: MicroserviceInfo
    unique_id_service_info: MicroserviceInfo
    media_service_info: MicroserviceInfo
    user_service_info: MicroserviceInfo
    url_shorten_service_info: MicroserviceInfo
    user_mention_service_info: MicroserviceInfo
    text_service_info: MicroserviceInfo
    post_storage_service_info: MicroserviceInfo
    user_timeline_service_info: MicroserviceInfo
    home_timeline_service_info: MicroserviceInfo
    social_graph_service_info: MicroserviceInfo


class ModuleConfiguration:
    """
    Make Configuration from conf_dict
    """

    # LoadBalancer
    basic_status_url: str = "basic_status"

    # Instance (Server) Configuration
    instance_type: str = conf_dict.get('Server').get('instance_type')
    init_number_of_instance: int = int(conf_dict.get('Server').get('number_of_instances'))
    image_id: str = conf_dict.get('Server').get('ami_id')
    snid: str = conf_dict.get('Server').get('snid')
    region_name: str = conf_dict.get('Server').get('region_name')
    az = conf_dict.get('Server').get('az')
    security_groups: str = conf_dict.get('Server').get('security_group')
    key_name: str = conf_dict.get('Server').get('key_name')
    server_tag_name = conf_dict.get('Server').get('tag_name')

    # Performance & Cost
    max_requests_per_sec_per_vm: int = int(conf_dict.get('Performance-Cost').get('max_requests_per_sec_per_vm'))
    slo = int(conf_dict.get('Performance-Cost').get('slo_criteria'))

    # Workload Configuration
    input_csv: str = conf_dict.get('Workload-Input').get('input_csv')
    trace_type = conf_dict.get('Workload-Input').get('trace_type')
    input_folder = f"{Path(__file__).parent.parent.absolute()}/workload_generator/{trace_type}/"
    input_csv_with_dir: str = os.path.join(input_folder, input_csv)
    workload_scaling_factor: int = 1
    duration: int = 60
    provisioning_method = conf_dict.get('Workload-Input').get('provisioning_method')
    over_provision: bool = pydantic.parse_obj_as(bool, conf_dict.get('Workload-Input').get('over_provision'))
    request_type = conf_dict.get('Workload-Input').get('request_type')
    use_case_for_experiment = conf_dict.get('Workload-Input').get('use_case_for_experiment')
    vm_affix = conf_dict.get('Workload-Input').get('vm_affix')
    lambda_url_for_invocation = conf_dict.get('Workload-Input').get('lambda_url_for_invocation')

    # External/Internal LoadBalancer Loadcat Addr
    external_compose_post_lb_addr: str = conf_dict.get('External-LoadBalancer').get('external-loadbalancer-addr')
    external_compose_post_lb_addr_private: str = conf_dict.get('External-LoadBalancer').get(
        'external-loadbalancer-addr-private')
    external_compose_post_lb_addr_with_loadcat_port: str = external_compose_post_lb_addr + ":26590"
    unique_id_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('unique-id-service-addr')
    media_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('media-service-addr')
    user_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('user-service-addr')
    url_shorten_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('url-shorten-service-addr')
    user_mention_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('user-mention-service-addr')
    text_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('text-service-addr')
    post_storage_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('post-storage-service-addr')
    user_timeline_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('user-timeline-service-addr')
    home_timeline_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('home-timeline-service-addr')
    social_graph_service_lb_addr: str = conf_dict.get('Internal-LoadBalancer').get('social-graph-service-addr')

    # External/Internal Loadbalancer Loadcat id
    external_compose_post_lb_id: str = conf_dict.get('External-LoadBalancer').get('compose-post-service-id')
    internal_unique_id_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('unique-id-service-id')
    media_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('media-service-id')
    user_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('user-service-id')
    url_shorten_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('url-shorten-service-id')
    user_mention_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('user-mention-service-id')
    text_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('text-service-id')
    post_storage_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('post-storage-service-id')
    user_timeline_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('user-timeline-service-id')
    home_timeline_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('home-timeline-service-id')
    social_graph_service_lb_id: str = conf_dict.get('Internal-LoadBalancer').get('social-graph-service-id')

    # Port
    compose_post_service_port = conf_dict.get('External-LoadBalancer').get('compose-post-service-port')
    unique_id_service_port = conf_dict.get('Internal-LoadBalancer').get('unique-id-service-port')
    media_service_port = conf_dict.get('Internal-LoadBalancer').get('media-service-port')
    user_service_port = conf_dict.get('Internal-LoadBalancer').get('user-service-port')
    url_shorten_service_port = conf_dict.get('Internal-LoadBalancer').get('url-shorten-service-port')
    user_mention_service_port = conf_dict.get('Internal-LoadBalancer').get('user-mention-service-port')
    text_service_port = conf_dict.get('Internal-LoadBalancer').get('text-service-port')
    post_storage_service_port = conf_dict.get('Internal-LoadBalancer').get('post-storage-service-port')
    user_timeline_service_port = conf_dict.get('Internal-LoadBalancer').get('user-timeline-service-port')
    home_timeline_service_port = conf_dict.get('Internal-LoadBalancer').get('home-timeline-service-port')
    social_graph_service_port = conf_dict.get('Internal-LoadBalancer').get('social-graph-service-port')

    # Lambda ARN
    external_compose_post_lambda_arn: str = conf_dict.get('External-LoadBalancer').get('compose-post-service-lambda')
    internal_unique_id_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('unique-id-service-lambda')
    media_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('media-service-lambda')
    user_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('user-service-lambda')
    url_shorten_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('url-shorten-service-lambda')
    user_mention_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('user-mention-service-lambda')
    text_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('text-service-lambda')
    post_storage_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('post-storage-service-lambda')
    user_timeline_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('user-timeline-service-lambda')
    home_timeline_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('home-timeline-service-lambda')
    social_graph_service_lambda_arn: str = conf_dict.get('Internal-LoadBalancer').get('social-graph-service-lambda')

    # File for compilation & Workload Result Log
    f_name: str = "resnet18_vm_for_test.py"
    result_logfile_folder: str = "logs/result"
    external_result_logfile_dir: str = result_logfile_folder + "/external_result.log"
    internal_result_logfile_dir: str = result_logfile_folder + "/internal_result.log"
    # Workload Result Log
    external_lb_logfile_dir: str = "logs/external_loadbalancer.log"
    internal_lb_logfile_dir: str = "logs/internal_loadbalancer.log"
    # Worker Log -> Keeps track of servers in Loadcat
    worker_log_dir: str = "logs/worker"
    external_compose_post_worker_file = "external_compose_post_worker.json"
    external_worker_log_file_name = "external_workers.json"
    external_worker_log_file_dir: str = worker_log_dir + "/" + external_worker_log_file_name
    internal_unique_id_service_worker_file = "internal_unique_id_service_worker.json"
    internal_media_service_worker_file = "internal_media_service_worker.json"
    internal_user_service_worker_file = "internal_user_service_worker.json"
    internal_url_shorten_service_worker_file = "internal_url_shorten_service_worker.json"
    internal_user_mention_service_worker_file = "internal_user_mention_service_worker.json"
    internal_text_service_worker_file = "internal_text_service_worker.json"
    internal_post_storage_worker_file = "internal_post_storage_worker.json"
    internal_user_timeline_worker_file = "internal_user_timeline_worker.json"
    internal_home_timeline_worker_file = "internal_home_timeline_worker.json"
    internal_social_graph_service_worker_file = "internal_social_graph_service_worker.json"
    internal_worker_log_file_name = "internal_workers.json"
    internal_worker_log_file_dir: str = worker_log_dir + "/" + internal_worker_log_file_name
    # Pickle Log
    pickle_dir = "logs/pickle"
    external_pickle_file_dir: str = pickle_dir + "/external_vm_instance.pkl"
    internal_pickle_file_dir: str = pickle_dir + "/internal_vm_instance.pkl"

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

    experiment_number: str = conf_dict.get('Workload-Input').get('experiment-number')


module_conf = ModuleConfiguration()


def gather_up_microservice_information():
    """
    Make information from social network service
    """
    global module_conf

    compose_post_service_info = MicroserviceInfo(service_name="compose-post-service",
                                                 service_worker_filename=module_conf.external_compose_post_worker_file,
                                                 service_port=module_conf.compose_post_service_port,
                                                 service_loadbalancer_id=module_conf.external_compose_post_lb_id,
                                                 service_loadbalancer_addr=module_conf.external_compose_post_lb_addr,
                                                 lambda_arn=module_conf.external_compose_post_lambda_arn)
    unique_id_service_info = MicroserviceInfo(service_name="unique-id-service",
                                              service_worker_filename=module_conf.internal_unique_id_service_worker_file,
                                              service_port=module_conf.unique_id_service_port,
                                              service_loadbalancer_id=module_conf.internal_unique_id_lb_id,
                                              service_loadbalancer_addr=module_conf.unique_id_lb_addr,
                                              lambda_arn=module_conf.internal_unique_id_lambda_arn)
    media_service_info = MicroserviceInfo(service_name="media-service",
                                          service_worker_filename=module_conf.internal_media_service_worker_file,
                                          service_port=module_conf.media_service_port,
                                          service_loadbalancer_id=module_conf.media_service_lb_id,
                                          service_loadbalancer_addr=module_conf.media_service_lb_addr,
                                          lambda_arn=module_conf.media_service_lambda_arn)
    user_service_info = MicroserviceInfo(service_name="user-service",
                                         service_worker_filename=module_conf.internal_user_service_worker_file,
                                         service_port=module_conf.user_service_port,
                                         service_loadbalancer_id=module_conf.user_service_lb_id,
                                         service_loadbalancer_addr=module_conf.user_service_lb_addr,
                                         lambda_arn=module_conf.user_service_lambda_arn)
    url_shorten_service_info = MicroserviceInfo(service_name="url-shorten-service",
                                                service_worker_filename=module_conf.internal_url_shorten_service_worker_file,
                                                service_port=module_conf.url_shorten_service_port,
                                                service_loadbalancer_id=module_conf.url_shorten_service_lb_id,
                                                service_loadbalancer_addr=module_conf.url_shorten_service_lb_addr,
                                                lambda_arn=module_conf.url_shorten_service_lambda_arn)
    user_mention_service_info = MicroserviceInfo(service_name="user-mention-service",
                                                 service_worker_filename=module_conf.internal_user_mention_service_worker_file,
                                                 service_port=module_conf.user_mention_service_port,
                                                 service_loadbalancer_id=module_conf.user_mention_service_lb_id,
                                                 service_loadbalancer_addr=module_conf.user_mention_service_lb_addr,
                                                 lambda_arn=module_conf.user_mention_service_lambda_arn)
    text_service_info = MicroserviceInfo(service_name="text-service",
                                         service_worker_filename=module_conf.internal_text_service_worker_file,
                                         service_port=module_conf.text_service_port,
                                         service_loadbalancer_id=module_conf.text_service_lb_id,
                                         service_loadbalancer_addr=module_conf.text_service_lb_addr,
                                         lambda_arn=module_conf.text_service_lambda_arn)
    post_storage_service_info = MicroserviceInfo(service_name="post-storage-service",
                                                 service_worker_filename=module_conf.internal_post_storage_worker_file,
                                                 service_port=module_conf.post_storage_service_port,
                                                 service_loadbalancer_id=module_conf.post_storage_service_lb_id,
                                                 service_loadbalancer_addr=module_conf.post_storage_service_lb_addr,
                                                 lambda_arn=module_conf.post_storage_service_lambda_arn)
    user_timeline_service_info = MicroserviceInfo(service_name="user-timeline-service",
                                                  service_worker_filename=module_conf.internal_user_timeline_worker_file,
                                                  service_port=module_conf.user_timeline_service_port,
                                                  service_loadbalancer_id=module_conf.user_timeline_service_lb_id,
                                                  service_loadbalancer_addr=module_conf.user_timeline_service_lb_addr,
                                                  lambda_arn=module_conf.user_timeline_service_lambda_arn)
    home_timeline_service_info = MicroserviceInfo(service_name="home-timeline-service",
                                                  service_worker_filename=module_conf.internal_home_timeline_worker_file,
                                                  service_port=module_conf.home_timeline_service_port,
                                                  service_loadbalancer_id=module_conf.home_timeline_service_lb_id,
                                                  service_loadbalancer_addr=module_conf.home_timeline_service_lb_addr,
                                                  lambda_arn=module_conf.home_timeline_service_lambda_arn)
    social_graph_service_info = MicroserviceInfo(service_name="social-graph-service",
                                                 service_worker_filename=module_conf.internal_social_graph_service_worker_file,
                                                 service_port=module_conf.social_graph_service_port,
                                                 service_loadbalancer_id=module_conf.social_graph_service_lb_id,
                                                 service_loadbalancer_addr=module_conf.social_graph_service_lb_addr,
                                                 lambda_arn=module_conf.social_graph_service_lambda_arn)

    return SocialNetworkInfo(compose_post_service_info, unique_id_service_info, media_service_info, user_service_info,
                             url_shorten_service_info, user_mention_service_info, text_service_info,
                             post_storage_service_info,
                             user_timeline_service_info, home_timeline_service_info, social_graph_service_info)


@dataclass
class ExperimentInfo:
    description: str
    lambda_url: str
    clientpool_suffix: str
    lambda_arn: str


def return_experiment_info():
    global module_conf

    experiment_info_dict = defaultdict()

    for exp_name, exp_description in conf_dict['Experiment-Description'].items():
        """
        Experiment-Description
        Experiment-Lambda-Arn
        Experiment-ClientPool-Suffix
        Experiment-Lambda-Url
        """
        lambda_arn = conf_dict['Experiment-Lambda-Arn'][exp_name]
        clientpool_suffix = conf_dict['Experiment-ClientPool-Suffix'][exp_name]
        clientpool_url = conf_dict['Experiment-Lambda-Url'][exp_name]

        experiment_info = ExperimentInfo(description=exp_description, lambda_url=clientpool_url,
                                         clientpool_suffix=clientpool_suffix, lambda_arn=lambda_arn)
        experiment_info_dict[exp_name] = experiment_info

    # pprint(dict(experiment_info_dict))
    # print(microservices_name_and_url_dict)

    return experiment_info_dict

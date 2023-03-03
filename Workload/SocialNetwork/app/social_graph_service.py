"""
Post Storage Service
"""
import datetime
import os
import random
import sys
from pathlib import Path
from typing import Union, List

from fastapi import FastAPI
from opentelemetry import trace

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))

# Import OpenTelemetry and Logger modules
from utils import utils, utils_opentelemetry, utils_social_network

# Mongo
from pymongo import MongoClient

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

test_input = utils_social_network.generate_post_class_input()

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
tracer = utils_opentelemetry.set_tracer()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()

# MongoClient
mongo_client = None
social_graph_collection = None

test_user_id = random.randint(1, 962)


def start_mongo_client_pool():
    global mongo_client
    global social_graph_collection

    mongo_port = 27017  # mongo_port = int(conf_dict.get('DB').get('social-graph-mongodb-port'))
    if mongo_client is None:
        mongo_client = MongoClient("social-graph-mongodb", mongo_port, waitQueueTimeoutMS=10000)
    social_graph_db = mongo_client['social_graph']
    social_graph_collection = social_graph_db['social_graph']


start_mongo_client_pool()


def GetFollowers(req_id: int, user_id: int, carrier: dict) -> List[int]:
    global logger
    global mongo_client, social_graph_collection

    parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    with tracer.start_as_current_span("GetFollowers", parent_ctx, kind=trace.SpanKind.SERVER):
        # logger.info(f"SocialGraphService Start {req_id} {utils.get_timestamp_ms()}")

        start_time = datetime.datetime.now()

        # Find follower and append user_id to result
        followers_user_id = []
        cursor = social_graph_collection.find(filter={'followees': user_id})
        for doc in cursor:
            follower_id = doc['user_id']
            followers_user_id.append(follower_id)

        # Insanity Check
        # cursor = social_graph_collection.find({})
        # for document in cursor:
        #     logger.debug(document)

        # # logger.info(f"{req_id} end post_storage_service {get_timestamp_ms()}")
        # # logger.info(f"SocialGraphService End {req_id} {utils.get_timestamp_ms()}")

        end_time = datetime.datetime.now()
        logger.info(f"SocialGraphService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
                    f" {(end_time - start_time).total_seconds()}")

        return followers_user_id


@app.get("/social_graph_service/{input_p}")
def run_social_graph_on_vm(input_p: Union[str, None] = None):
    """
    Run Social Graph Service on VM
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # Parameter
    req_id = parsed_inputs.get('req_id', 3)
    user_id = parsed_inputs.get('user_id', 4)
    carrier = parsed_inputs.get('carrier', {})

    # Call UserMentionService
    # logger.info(f"Call SocialGraphService Start {req_id} {utils.get_timestamp_ms()}")
    res = GetFollowers(req_id, user_id, carrier)

    # social_graph_res = native_object_encoded(res)
    # return social_graph_res

    return res

#
# @app.on_event("startup")
# async def startup_event():
#     """
#     Init logger and config.ini
#     """

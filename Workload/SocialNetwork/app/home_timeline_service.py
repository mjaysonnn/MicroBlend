import datetime
import os
import sys
from pathlib import Path
from typing import Union, List

import requests
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))

# Import OpenTelemetry and Logger modules
from utils import utils, utils_opentelemetry, utils_social_network

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
tracer = utils_opentelemetry.set_tracer()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()

# test input for local testing
post_parameters = utils_social_network.generate_post_class_input()
user_mention_id_list = list()
user_mentions = post_parameters.user_mentions
for each_user_mention in user_mentions:
    user_mention_id_list.append(each_user_mention.user_id)

# Mongo
import pymongo
from pymongo import MongoClient

# MongoClient
mongo_client = None
home_timeline_collection = None


def start_mongo_client_pool():
    global logger
    global mongo_client
    global home_timeline_collection

    # Init MongoClient
    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("home-timeline-mongodb", mongo_port, waitQueueTimeoutMS=10000)
    post_db = mongo_client['home_timeline']
    home_timeline_collection = post_db['home_timeline']

    # Create index
    home_timeline_collection.create_index([('post_id', pymongo.ASCENDING)], name='post_id', unique=True)

    # logger.debug(home_timeline_collection.index_information())


start_mongo_client_pool()


def invoke_social_graph_service(req_id, user_id, carrier):
    """
    Invoke SocialGraphService
    """

    # logger.info(f"WriteHomeTimeline Invoke SocialGraphService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "user_id": user_id, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    # url_shorten_service is container name
    url = f"http://social-graph-service:5011/social_graph_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200
    res = resp.json()

    # # logger.info(f"WriteHomeTimeline Invoke SocialGraphService End {req_id} {utils.get_timestamp_ms()}")
    return res


def WriteHomeTimeline(req_id: int, post_id: int, user_id: int, timestamp: int, user_mentions_id: List[int],
                      carrier: dict) -> None:
    global logger
    global mongo_client, home_timeline_collection

    # Start OpenTelemetry Tracer - If there is parent context, use it
    parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    with tracer.start_as_current_span("WriteHomeTimeline", parent_ctx, kind=trace.SpanKind.SERVER):
        start_time = datetime.datetime.now()

        home_timeline_ids = []

        # Inject OpenTelemetry Context to carrier
        TraceContextTextMapPropagator().inject(carrier)

        # Invoke GetFollowers from SocialGraphService
        social_graph_result: List[int] = invoke_social_graph_service(req_id, user_id, carrier)

        # Append results to home_timeline_ids
        home_timeline_ids.extend(social_graph_result)

        # Include user mentions in result
        for user_mention in user_mentions_id:
            home_timeline_ids.append(user_mention)

        # # Update Post id and Timestamp of User id
        for home_timeline_id in home_timeline_ids:
            post_to_insert = {'post_id': post_id, 'timestamp': timestamp, 'home_timeline_id': home_timeline_id}

            home_timeline_collection.update_one({'post_id': post_to_insert.get('post_id')}, {'$set': post_to_insert},
                                                upsert=True)

        # ## Insanity Check
        # cursor = home_timeline_collection.find({})
        # for document in cursor:
        #     logger.debug(document)

        # # logger.info(f"{req_id} end home_timeline_service {get_timestamp_ms()}")
        # # logger.info(f"HomeTimelineService End {req_id} {utils.get_timestamp_ms()}")
        end_time = datetime.datetime.now()
        logger.info(f"HomeTimelineService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
                    f" {(end_time - start_time).total_seconds()}")


@app.get("/home_timeline_service/{input_p}")
def run_home_timeline_service_on_vm(input_p: Union[str, None] = None):
    """
    HomeTimelineService
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # 2. Parameters
    req_id = parsed_inputs.get('req_id', 3)
    post_id = parsed_inputs.get('post_id', post_parameters.post_id)
    user_id = parsed_inputs.get('user_id', 3)
    timestamp = parsed_inputs.get('timestamp', utils.get_timestamp_ms())
    user_mentions_id = parsed_inputs.get('user_mentions_id', user_mention_id_list)
    carrier = parsed_inputs.get('carrier', {})

    # 3. Invoke WriteHomeTimeline
    # logger.info(f"Call HomeTimelineService Start {req_id} {utils.get_timestamp_ms()}")
    WriteHomeTimeline(req_id, post_id, user_id, timestamp, user_mentions_id, carrier)
    # # logger.info(f"Call HomeTimelineService End {req_id} {get_timestamp_ms()}")

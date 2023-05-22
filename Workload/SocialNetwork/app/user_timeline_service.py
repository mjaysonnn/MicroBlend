import datetime
import os
import random
import sys
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))

# Import OpenTelemetry and Logger modules
from utils import utils, utils_social_network

# Mongo
import pymongo
from pymongo import MongoClient

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer


# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()

# MongoClient
mongo_client = None
user_timeline_collection = None

test_user_id = random.randint(1, 962)
test_input = utils_social_network.generate_post_class_input()


def start_mongo_client_pool():
    global logger
    global mongo_client
    global user_timeline_collection

    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("user-timeline-mongodb", waitQueueTimeoutMS=10000)

    # Access post collection (Table)
    user_timeline_db = mongo_client['user_timeline']
    user_timeline_collection = user_timeline_db['user_timeline']

    # Index
    user_timeline_collection.create_index([('user_id', pymongo.ASCENDING)], name='user_id', unique=True)


start_mongo_client_pool()


def WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier):
    global logger
    global mongo_client, user_timeline_collection

    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("WriteUserTimeline", parent_ctx, kind=trace.SpanKind.SERVER):
    # start_time = utils.get_timestamp_ms()
    start_time = datetime.datetime.now()

    # logger.info(f"UserTimelineService Start {req_id} {utils.get_timestamp_ms()}")

    user_timeline_collection.find_one_and_update(filter={'user_id': user_id},
                                                 update={'$push': {'posts': {'post_id': post_id,
                                                                             'timestamp': timestamp
                                                                             }
                                                                   }},
                                                 upsert=True)

    # Insanity Check
    # cursor = user_timeline_collection.find({})
    # for document in cursor:
    #     logger.debug(document)

    end_time = datetime.datetime.now()
    # logger.info(f"UserTimelineService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")


@app.get("/user_timeline_service/{input_p}")
def run_user_timeline_service_on_vm(input_p: Union[str, None] = None):
    """
    Run Social Graph Service on VM
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # 2. Parameter
    req_id = parsed_inputs.get('req_id', 3)
    post_id = parsed_inputs.get('post_id', test_input.post_id)
    user_id = parsed_inputs.get('user_id', test_user_id)
    timestamp = parsed_inputs.get('timestamp', utils.get_timestamp_ms())
    carrier = parsed_inputs.get('carrier', {})

    # 3. Call UserMentionService
    # logger.info(f"Call SocialGraphService Start {req_id} {utils.get_timestamp_ms()}")
    WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier)

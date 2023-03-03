"""Parse and generate input for ComposePostService
Modules
 =========

 1. `Init Logger`
 2. `Fetch config.ini`
 3. `Receive Public/Private IP`
 4. `Parse Arguments / If not (Test), make manual input`
 5. UserMentionService
 6. `Return Unique ID`

"""
import datetime
import os
import sys
from pathlib import Path
from typing import Union, List

from fastapi import FastAPI
from opentelemetry import trace

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

# Import OpenTelemetry and Logger modules
from utils import utils, utils_opentelemetry

# Mongo
import pymongo
from pymongo import MongoClient

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

user_name_list = ["username_1", "username_2"]

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
user_collection = None


def start_mongo_client_pool():
    global logger
    global mongo_client
    global user_collection

    # MongoClient
    mongodb_ip_addr = "social-graph-mongodb"
    mongodb_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient(mongodb_ip_addr, mongodb_port, waitQueueTimeoutMS=10000)

    user_db = mongo_client['user']
    user_collection = user_db['user']

    # Index
    user_collection.create_index([('username', pymongo.ASCENDING)], name='username', unique=True)


start_mongo_client_pool()


def ComposeUserMentions(req_id: int, usernames: List[str], carrier: dict) -> List[UserMention]:
    global logger
    global user_collection

    parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    with tracer.start_as_current_span("ComposeUserMentions", parent_ctx, kind=trace.SpanKind.SERVER):
        start_time = datetime.datetime.now()
        # logger.info(f"UserMentionService Start {req_id} {utils.get_timestamp_ms()}")

        user_mention_list = list()
        for each_username in usernames:
            # find user_id from mongodb collection
            post = user_collection.find_one(filter={'username': each_username})
            user_id = post.get("user_id")
            user_mention_list.append(UserMention(user_id, each_username))

        end_time = datetime.datetime.now()
        logger.info(f"UserMentionService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
                    f" {(end_time - start_time).total_seconds()}")

        return user_mention_list


@app.get("/user_mention_service/{input_p}")
def run_user_mention_service_on_vm(input_p: Union[str, None] = None):
    # logger.debug("input_p: {}".format(input_p))

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # Parameter
    req_id = parsed_inputs.get('req_id', 3)
    usernames = parsed_inputs.get('usernames', user_name_list)
    carrier = parsed_inputs.get('carrier', {})

    # Call UserMentionService
    user_mention_res = ComposeUserMentions(req_id=req_id, usernames=usernames, carrier=carrier)

    return user_mention_res

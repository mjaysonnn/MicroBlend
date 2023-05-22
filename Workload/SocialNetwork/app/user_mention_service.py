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
from fastapi import FastAPI
from pathlib import Path
from typing import Union, List

# from opentelemetry import trace

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

# Import OpenTelemetry and Logger modules
from utils import utils

# Mongo
import pymongo
from pymongo import MongoClient

# from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

user_name_list = ["username_1", "username_2"]

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
# tracer = utils_opentelemetry.set_tracer()

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
    # # logger.info(f"MongoClient Connected to {mongodb_ip_addr}:{mongodb_port}")
    # # logger.info(f"MongoClient List of Databases: {mongo_client.list_database_names()}")
    # # logger.info(mongo_client)
    # Access to collection (Table)
    user_db = mongo_client['user']
    user_collection = user_db['user']

    # Index
    user_collection.create_index([('username', pymongo.ASCENDING)], name='username', unique=True)


start_mongo_client_pool()


def ComposeUserMentions(req_id: int, usernames: List[str], carrier: dict) -> List[UserMention]:
    global logger
    global user_collection

    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("ComposeUserMentions", parent_ctx, kind=trace.SpanKind.SERVER):

    start_time = datetime.datetime.now()
    # logger.info(f"UserMentionService Start {req_id} {utils.get_timestamp_ms()}")

    user_mention_list = list()
    for each_username in usernames:
        # find user_id from mongodb collection
        post = user_collection.find_one(filter={'username': each_username})
        user_id = post.get("user_id")
        user_mention_list.append(UserMention(user_id, each_username))

    # # logger.info(f"UserMentionService End {req_id} {utils.get_timestamp_ms()}")

    end_time = datetime.datetime.now()
    # logger.info(f"UserMentionService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")

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
    # logger.debug("req_id: {}".format(req_id))
    # logger.debug("usernames: {}".format(usernames))
    # logger.debug("carrier: {}".format(carrier))

    # Call UserMentionService
    # logger.info(f"Call UserMentionService Start {req_id} {utils.get_timestamp_ms()}")
    user_mention_res = ComposeUserMentions(req_id=req_id, usernames=usernames, carrier=carrier)
    # # logger.info(f"Call UserMentionService End {req_id} {utils.get_timestamp_ms()}")

    # encoded_user_mention_res = native_object_encoded(user_mention_res)
    # return encoded_user_mention_res

    return user_mention_res

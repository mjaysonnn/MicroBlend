import datetime
import os
import random
import sys
from pathlib import Path
from typing import Union

from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))

from utils import utils, utils_social_network

# Mongo
import pymongo
from pymongo import MongoClient

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# MongoClient
mongo_client = None
user_timeline_collection = None

test_user_id = random.randint(1, 962)
test_input = utils_social_network.generate_post_class_input()


def start_mongo_client_pool():
    global mongo_client
    global user_timeline_collection

    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("user-timeline-mongodb", waitQueueTimeoutMS=10000)

    # Access post collection (Table)
    user_timeline_db = mongo_client["user_timeline"]
    user_timeline_collection = user_timeline_db["user_timeline"]

    # Index
    user_timeline_collection.create_index(
        [("user_id", pymongo.ASCENDING)], name="user_id", unique=True
    )


start_mongo_client_pool()


def WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier):
    global mongo_client, user_timeline_collection

    start_time = datetime.datetime.now()

    user_timeline_collection.find_one_and_update(
        filter={"user_id": user_id},
        update={"$push": {"posts": {"post_id": post_id, "timestamp": timestamp}}},
        upsert=True,
    )

    # Insanity Check
    # cursor = user_timeline_collection.find({})
    # for document in cursor:
    #     logger.debug(document)

    end_time = datetime.datetime.now()


@app.get("/user_timeline_service/{input_p}")
def run_user_timeline_service_on_vm(input_p: Union[str, None] = None):
    """
    Run Social Graph Service on VM
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Parameter
    req_id = parsed_inputs.get("req_id", 3)
    post_id = parsed_inputs.get("post_id", test_input.post_id)
    user_id = parsed_inputs.get("user_id", test_user_id)
    timestamp = parsed_inputs.get("timestamp", utils.get_timestamp_ms())
    carrier = parsed_inputs.get("carrier", {})

    WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier)

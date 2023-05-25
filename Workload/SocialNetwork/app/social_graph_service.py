import datetime
import os
import random
import sys
from pathlib import Path
from typing import Union, List

from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))

from utils import utils, utils_social_network

# Mongo
from pymongo import MongoClient

test_input = utils_social_network.generate_post_class_input()

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# MongoClient
mongo_client = None
social_graph_collection = None

test_user_id = random.randint(1, 962)


def start_mongo_client_pool():
    global mongo_client
    global social_graph_collection

    if mongo_client is None:
        mongo_port = 27017
        mongo_client = MongoClient(
            "social-graph-mongodb", mongo_port, waitQueueTimeoutMS=10000
        )
    social_graph_db = mongo_client["social_graph"]
    social_graph_collection = social_graph_db["social_graph"]


start_mongo_client_pool()


def GetFollowers(req_id: int, user_id: int, carrier: dict) -> List[int]:
    global mongo_client, social_graph_collection

    start_time = datetime.datetime.now()

    # Find follower and append user_id to result
    followers_user_id = []
    cursor = social_graph_collection.find(filter={"followees": user_id})
    followers_user_id.extend(doc["user_id"] for doc in cursor)

    end_time = datetime.datetime.now()

    return followers_user_id


@app.get("/social_graph_service/{input_p}")
def run_social_graph_on_vm(input_p: Union[str, None] = None):
    """
    Run Social Graph Service on VM
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    req_id = parsed_inputs.get("req_id", 3)
    user_id = parsed_inputs.get("user_id", 4)
    carrier = parsed_inputs.get("carrier", {})

    return GetFollowers(req_id, user_id, carrier)

#

import datetime
import os
import sys
from pathlib import Path
from typing import Union, List

import requests
from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))

from utils import utils, utils_social_network

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# test input for local testing
post_parameters = utils_social_network.generate_post_class_input()
user_mentions = post_parameters.user_mentions
user_mention_id_list = [
    each_user_mention.user_id for each_user_mention in user_mentions
]
# Mongo
import pymongo
from pymongo import MongoClient

# MongoClient
mongo_client = None
home_timeline_collection = None


def start_mongo_client_pool():
    """
    Start MongoClient
    """

    global mongo_client
    global home_timeline_collection

    if mongo_client is None:
        mongo_port = 27017
        mongo_client = MongoClient(
            "home-timeline-mongodb", mongo_port, waitQueueTimeoutMS=10000
        )
    post_db = mongo_client["home_timeline"]
    home_timeline_collection = post_db["home_timeline"]

    # Create index
    home_timeline_collection.create_index(
        [("post_id", pymongo.ASCENDING)], name="post_id", unique=True
    )


start_mongo_client_pool()


def invoke_social_graph_service(req_id, user_id, carrier):
    """
    Invoke SocialGraphService
    """

    input_d = {"req_id": req_id, "user_id": user_id, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://social-graph-service:5011/social_graph_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200
    return resp.json()


def WriteHomeTimeline(
        req_id: int,
        post_id: int,
        user_id: int,
        timestamp: int,
        user_mentions_id: List[int],
        carrier: dict,
) -> None:
    global mongo_client, home_timeline_collection

    start_time = datetime.datetime.now()

    # Invoke GetFollowers from SocialGraphService
    social_graph_result: List[int] = invoke_social_graph_service(
        req_id, user_id, carrier
    )

    home_timeline_ids = list(social_graph_result)

    # Include user mentions in result
    home_timeline_ids.extend(iter(user_mentions_id))
    # # Update Post id and Timestamp of User id
    for home_timeline_id in home_timeline_ids:
        post_to_insert = {
            "post_id": post_id,
            "timestamp": timestamp,
            "home_timeline_id": home_timeline_id,
        }

        home_timeline_collection.update_one(
            {"post_id": post_to_insert.get("post_id")},
            {"$set": post_to_insert},
            upsert=True,
        )

    end_time = datetime.datetime.now()


@app.get("/home_timeline_service/{input_p}")
def run_home_timeline_service_on_vm(input_p: Union[str, None] = None):
    """
    HomeTimelineService
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Parameters
    req_id = parsed_inputs.get("req_id", 3)
    post_id = parsed_inputs.get("post_id", post_parameters.post_id)
    user_id = parsed_inputs.get("user_id", 3)
    timestamp = parsed_inputs.get("timestamp", utils.get_timestamp_ms())
    user_mentions_id = parsed_inputs.get("user_mentions_id", user_mention_id_list)
    carrier = parsed_inputs.get("carrier", {})

    # 3. Invoke WriteHomeTimeline

    WriteHomeTimeline(req_id, post_id, user_id, timestamp, user_mentions_id, carrier)

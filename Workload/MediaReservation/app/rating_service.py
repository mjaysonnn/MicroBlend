import datetime
from pathlib import Path
from typing import Union

from fastapi import FastAPI
from pymongo import MongoClient

from utils import utils

app = FastAPI()

logger = utils.init_logger(Path(__file__).parent.absolute())

# MongoClient
mongo_client = None
rating_collection = None


def start_mongo_client_pool():
    """
    Start MongoClient Pool"""
    global logger
    global mongo_client
    global rating_collection

    if mongo_client is None:
        mongo_port = 27017
        mongo_client = MongoClient(
            "rating-mongodb", mongo_port, waitQueueTimeoutMS=10000
        )
    post_db = mongo_client["rating"]
    rating_collection = post_db["rating"]


start_mongo_client_pool()


def UploadRating(req_id, movie_id, rating):
    """
    Compose Creator with User ID for PostService
    """

    start_time = datetime.datetime.now()

    user_id = str(req_id) + str(int(datetime.datetime.now().timestamp()))

    post_to_insert = {
        "user_id": user_id,
        "rating": rating,
    }

    rating_collection.update_one(
        {"user_id": post_to_insert.get("user_id")},
        {"$set": post_to_insert},
        upsert=True,
    )

    end_time = datetime.datetime.now()


@app.get("/rating_service/{input_p}")
def run_rating_service(input_p: Union[str, None] = None):
    """
    Run rating service
    """
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Extract the inputs
    req_id = parsed_inputs.get("req_id", 3)
    movie_id = parsed_inputs.get("movie_id", "abc")
    rating = parsed_inputs.get("rating", 3)

    # 3. Call the service
    UploadRating(req_id, movie_id, rating)

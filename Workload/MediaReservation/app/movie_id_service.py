import datetime
import random
import string
from pathlib import Path
from typing import Union

import requests
from fastapi import FastAPI
from pymongo import MongoClient

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# MongoClient
mongo_client = None
movie_id_collection = None


def start_mongo_client_pool():
    global logger
    global mongo_client
    global movie_id_collection

    # Init MongoClient
    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("movie-id-mongodb", mongo_port, waitQueueTimeoutMS=10000)
    post_db = mongo_client['movie-id']
    movie_id_collection = post_db['movie-id']

    # Create index
    # movie_id_collection.create_index([('movie-id', pymongo.ASCENDING)], name='post_id', unique=True)


start_mongo_client_pool()


def invoke_rating_service(req_id, movie_id, rating):
    """
        Invoke another service"""

    # generate random string of 5 characters

    input_d = {"req_id": req_id, "movie_id": movie_id, "rating": rating}

    encoded_input = utils.native_object_encoded(input_d)

    # url_shorten_service is container name
    url = f"http://rating-service:5004/rating_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200
    res = resp.json()


def UploadMovieId(req_id, title, rating):
    """
    Compose Creator with User ID for PostService
    """

    start_time = datetime.datetime.now()

    movie_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    post_to_insert = {
        'movie_id': movie_id,
        'title': title
    }
    # logger.debug(title)
    # logger.debug(movie_id)

    movie_id_collection.update_one({'movie_id': post_to_insert.get('movie_id')}, {'$set': post_to_insert}, upsert=True)

    # logger.debug(movie_id)
    # logger.debug(title)

    invoke_rating_service(req_id, movie_id, rating)

    end_time = datetime.datetime.now()
    # logger.info(f"UploadMovieId {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")

    return movie_id


@app.get("/movie_id_service/{input_p}")
def run_movie_id_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Extract the inputs
    req_id = parsed_inputs.get("req_id", 1232)
    title = parsed_inputs.get("title", "abc")
    rating = parsed_inputs.get("rating", 3)

    # 3. Call the service
    movie_id = UploadMovieId(req_id, title, rating)
    return movie_id

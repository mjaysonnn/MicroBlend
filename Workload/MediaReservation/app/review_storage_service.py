import dataclasses
import datetime
import random
import string
from pathlib import Path
from typing import Union

from fastapi import FastAPI
from pymongo import MongoClient

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini


# MongoClient
mongo_client = None
review_storage_collection = None


@dataclasses.dataclass
class Review:
    review_id: int
    user_id: int
    req_id: int
    text: str
    movie_id: str
    rating: int
    timestamp: int


def generateReviewParameter(req_id: int = None):
    """
    Generate ComposeReviewParameter input for ComposePostService
    """
    # req_id, user_id, username
    if req_id is None:
        req_id = random.getrandbits(63)

    # Text -> add user mention and url
    text = ''.join(random.choices(string.ascii_letters + string.digits, k=100))

    # generate rest of the elements in Review
    movie_id = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    rating = random.randint(1, 5)
    # timestamp to float
    timestamp = int(datetime.datetime.now().timestamp())
    user_id = random.getrandbits(63)
    review_id = random.getrandbits(63)

    # Generate ComposerReviewParameter Instance
    compose_review_parameter = Review(review_id=review_id, user_id=user_id, req_id=req_id, text=text, movie_id=movie_id,
                                      rating=rating, timestamp=timestamp)

    return compose_review_parameter


def start_mongo_client_pool():
    global logger
    global mongo_client
    global review_storage_collection

    # Init MongoClient
    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("review-storage-mongodb", mongo_port, waitQueueTimeoutMS=10000)
    post_db = mongo_client['review']
    review_storage_collection = post_db['review']

    # insanity check
    # logger.debug(review_storage_collection)


start_mongo_client_pool()


def StoreReview(req_id, review):
    """
    Compose Creator with User ID for PostService
    """
    global logger

    start_time = datetime.datetime.now()

    post_to_insert = {'review_id': review.review_id,
                      'user_id': review.user_id,
                      'req_id': review.req_id,
                      'text': review.text,
                      'movie_id': review.movie_id,
                      'rating': review.rating,
                      'timestamp': review.timestamp}

    # logger.debug(f"post_to_insert: {post_to_insert}")

    review_storage_collection.update_one({'review_id': post_to_insert.get('review_id')}, {'$set': post_to_insert},
                                         upsert=True)

    end_time = datetime.datetime.now()
    # logger.info(f"StoreReview {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")


@app.get("/review_storage_service/{input_p}")
def run_store_review_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug(f"parsed_inputs: {parsed_inputs}")
    # 2. Extract the inputs
    req_id = parsed_inputs.get('req_id', random.getrandbits(3))
    # logger.debug(f"req_id: {req_id}")

    review = parsed_inputs.get('review', None)
    if review is None:
        review = generateReviewParameter()

    # 3. Run the service
    StoreReview(req_id, review)

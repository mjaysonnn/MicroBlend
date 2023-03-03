import dataclasses
import datetime
import random
import string
from pathlib import Path
from typing import Union

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
user_review_collection = None


def start_mongo_client_pool():
    global logger
    global mongo_client
    global user_review_collection

    # Init MongoClient
    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("user-review-mongodb", mongo_port, waitQueueTimeoutMS=10000)
    post_db = mongo_client['user-review']
    user_review_collection = post_db['user-review']


start_mongo_client_pool()


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


review_parameter = generateReviewParameter()


def UploadUserReview(req_id, user_id, review_id, timestamp):
    """
    Compose Creator with User ID for PostService
    """
    global logger

    start_time = datetime.datetime.now()

    post_to_insert = {'user_id': user_id, 'review_id': review_id, 'timestamp': timestamp}

    user_review_collection.update_one({'review_id': post_to_insert.get('review_id')}, {'$set': post_to_insert},
                                      upsert=True)

    end_time = datetime.datetime.now()
    # logger.info(f"UploadUserReview {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")


@app.get("/user_review_service/{input_p}")
def run_user_review_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Extract the inputs
    req_id = parsed_inputs.get('req_id', 1212121)

    user_id = parsed_inputs.get('user_id', review_parameter.user_id)
    review_id = parsed_inputs.get('review_id', review_parameter.review_id)
    timestamp = parsed_inputs.get('timestamp', review_parameter.timestamp)

    # 3. Run the service
    UploadUserReview(req_id, user_id, review_id, timestamp)
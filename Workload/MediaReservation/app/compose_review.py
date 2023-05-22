import dataclasses
import datetime
import random
import requests
import string
from concurrent.futures import ThreadPoolExecutor, wait
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# Import OpenTelemetry and Logger modules
from utils import utils

app = FastAPI()  # Init FastAPI Application

logger = utils.init_logger(Path(__file__).parent.absolute())  # Logging to file


@dataclasses.dataclass
class ComposeReviewParameter:
    req_id: int = None
    text: str = None
    title: str = None
    rating: int = None
    first_name: str = None
    last_name: str = None
    username: str = None
    password: str = None


def generate_input_for_compose_review(req_id=None):
    """
    Generate ComposeReviewParameter input for ComposePostService
    """
    # req_id, user_id, username
    if req_id is None:
        req_id = random.getrandbits(63)

    # Text -> add user mention and url
    text = ''.join(random.choices(string.ascii_letters + string.digits, k=100))

    # Make title rating first_name last_name username password
    title = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    rating = random.randint(1, 5)
    first_name = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    last_name = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    username = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

    # Generate ComposerReviewParameter Instance
    compose_review_parameter = ComposeReviewParameter(req_id=req_id, text=text, title=title, rating=rating,
                                                      first_name=first_name, last_name=last_name, username=username,
                                                      password=password)

    return compose_review_parameter


# make Review dataclass that consists of review_id, user_id, req_id, text, movie_id, rating, timestamp
@dataclasses.dataclass
class Review:
    review_id: int
    user_id: int
    req_id: int
    text: str
    movie_id: str
    rating: int
    timestamp: int


def invoke_text_service(req_id, text):
    """
    # 1. Get Text from TextService
    """
    # logger.info(f"ComposePostService Invoke TextService Start {req_id}  {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "text": text}
    # logger.debug(input_d)
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://text-service:5002/text_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    text_res = resp.json()
    # logger.debug(text_res)
    # get result from text service

    return text_res


def invoke_upload_movie_id_service(req_id, title, rating):
    input_d = {"req_id": req_id, "title": title, "rating": rating}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://movie-id-service:5005/movie_id_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200

    movie_id = resp.json()
    return movie_id


def invoke_unique_id_service(req_id):
    """
    3. Invoke MediaService
    """
    # logger.info(f"ComposePostService Invoke MediaService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://unique-id-service:5001/unique_id_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    unique_id = resp.json()

    return unique_id


def invoke_user_service(req_id, first_name, last_name, username, password):
    """
    4. Invoke UniqueIdService
    """
    # logger.info(f"ComposePostService Invoke UniqueIdService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "first_name": first_name, "last_name": last_name,
               "username": username, "password": password, }

    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://user-service:5003/user_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    user_id = resp.json()
    return user_id


def invoke_review_storage_service(req_id, review):
    input_d = {"req_id": req_id, "review": review}
    encoded_input = utils.native_object_encoded(input_d)
    # logger.debug(input_d)
    # logger.debug(encoded_input)
    url = f"http://review-storage-service:5006/review_storage_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    # logger.debug(resp.json())


# 6. WriteUserTimeline from UserTimelineService
def invoke_user_review_service(req_id, review_id, user_id, timestamp):
    input_d = {"req_id": req_id, "review_id": review_id, "user_id": user_id, "timestamp": timestamp}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://user-review-service:5008/user_review_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


def invoke_movie_review_service(req_id, movie_id, review_id, timestamp) -> None:
    """
    7. Invoke HomeTimelineService
    """

    input_d = {"req_id": req_id, "movie_id": movie_id, "review_id": review_id, "timestamp": timestamp}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://movie-review-service:5007/movie_review_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


def ComposeAndUpload(req_id, text, title, rating, first_name, last_name, username, password):
    """
    Invoke and Make Review and invoke 3 Microservices
    """
    # Extract URLs from text

    # start_time = utils.get_timestamp_ms()
    start_time = datetime.datetime.now()

    # insanity check
    if not text or not title or not rating or not first_name or not last_name or not username or not password:
        return

    with ThreadPoolExecutor() as executor:  # Invoke Microservices to get variables for Post
        text_future = executor.submit(invoke_text_service, req_id, text)
        upload_movie_id_future = executor.submit(invoke_upload_movie_id_service, req_id, title, rating)
        unique_id_future = executor.submit(invoke_unique_id_service, req_id)
        user_future = executor.submit(invoke_user_service, req_id, first_name, last_name, username, password)

        review_id = unique_id_future.result()  # Result from MediaService
        # logger.debug(f"review_id: {review_id}")

        text = text_future.result()  # Result from TextService
        # logger.debug(f"text: {text}")

        user_id = user_future.result()  # Result from UniqueIdService
        # logger.debug(f"user_id: {user_id}")

        movie_id = upload_movie_id_future.result()  # Result from UserService
        # logger.debug(f"movie_id: {movie_id}")

    # Make Review
    timestamp = utils.get_timestamp_ms()
    review = Review(review_id, user_id, req_id, text, movie_id, rating, timestamp)
    # logger.debug(review)

    # Invoke Microservices
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(invoke_review_storage_service, req_id, review),
                   executor.submit(invoke_user_review_service, req_id, review.user_id,
                                   review.review_id, timestamp),
                   executor.submit(invoke_movie_review_service, req_id, review.movie_id,
                                   review.review_id, timestamp)]
        wait(futures)

    end_time = datetime.datetime.now()
    # print the response time in milliseconds
    # logger.info(f"UploadAndCompose {req_id} {start_time} {end_time} {(end_time - start_time).total_seconds()}")
    logger.info(f"UploadAndCompose {req_id} {start_time.timestamp()} {end_time.timestamp()}"
                f" {(end_time - start_time).total_seconds()*1000}")


@app.get("/compose_review_service/{input_p}")
def run_compose_post_service_on_vm(input_p: Union[str, None] = None):
    """
    Parse and generate input for ComposePostService
    :param input_p:
    :return:
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # Parameters
    compose_post_parameter = generate_input_for_compose_review()
    req_id = parsed_inputs.get("req_id", compose_post_parameter.req_id)
    text = parsed_inputs.get("text", compose_post_parameter.text)
    title = parsed_inputs.get("title", compose_post_parameter.title)
    rating = parsed_inputs.get("rating", compose_post_parameter.rating)
    first_name = parsed_inputs.get("first_name", compose_post_parameter.first_name)
    last_name = parsed_inputs.get("last_name", compose_post_parameter.last_name)
    username = parsed_inputs.get("username", compose_post_parameter.username)
    password = parsed_inputs.get("password", compose_post_parameter.password)

    carrier = {}

    # logger.info(f"ComposePostService {req_id} {text} {title} {rating} {first_name} {last_name} {username} {password}")

    # 2. Invoke ComposeAndUpload
    ComposeAndUpload(req_id, text, title, rating, first_name, last_name, username, password)

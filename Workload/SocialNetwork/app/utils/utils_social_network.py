import os
import random
import string
import sys
from dataclasses import field, dataclass
from typing import List

path = os.getcwd()

parent = os.path.dirname(path)

sys.path.append(os.path.join(parent, "gen-py"))

from social_network.ttypes import *


def get_random_string(length):
    """
    Generate a random string of fixed length

    """
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


# This is used for HomeTimelineService and UserTimelineService
def generate_post_class_input(req_id=None):
    """
    Generate Post class input for HomeTimelineService and UserTimelineService and PostStorageService
    """
    if req_id is None:
        req_id = random.getrandbits(63)

    text = "HelloWorld"

    media_0 = Media(media_id=0, media_type="png")
    media_1 = Media(media_id=1, media_type="png")
    media = [media_0, media_1]

    post_id = 0
    post_type = PostType.POST

    creator = Creator(username="user_0", user_id=0)

    url_0 = Url(shortened_url="shortened_url_0", expanded_url="expanded_url_0")
    url_1 = Url(shortened_url="shortened_url_1", expanded_url="expanded_url_1")
    urls = [url_0, url_1]

    user_mention_0 = UserMention(user_id=1, username="username_1")
    user_mention_1 = UserMention(user_id=2, username="username_2")
    user_mentions = [user_mention_0, user_mention_1]

    return Post(
        user_mentions=user_mentions,
        req_id=req_id,
        creator=creator,
        post_type=post_type,
        urls=urls,
        media=media,
        post_id=post_id,
        text=text,
    )


@dataclass
class ComposePostServiceParameters:
    req_id: int = None
    username: str = None
    user_id: int = 1
    text: str = None
    media_ids: List = field(default_factory=list)
    media_types: list = field(default_factory=list)
    post_type: int = 0


def generate_input_for_compose_post_service(req_id=None):
    """
    Generate ComposePostServiceParameters input for ComposePostService
    """
    # req_id, user_id, username
    if req_id is None:
        req_id = random.getrandbits(63)
    user_id = random.randint(1, 962)
    username = f"username_{user_id}"

    # Text -> add user mention and url
    text = "".join(random.choices(string.ascii_letters + string.digits, k=100))
    user_mention_ids = []
    num_user_mentions = random.randint(0, 3)
    for _ in range(num_user_mentions):
        while True:
            user_mention_id = random.randint(1, 962)
            if user_mention_id != user_id and user_mention_id not in user_mention_ids:
                user_mention_ids.append(user_mention_id)
                break
    for user_mention_id in user_mention_ids:
        text = f"{text} @username_{str(user_mention_id)}"
    num_urls = random.randint(0, 3)
    for _ in range(num_urls):
        text = f"{text} http://{get_random_string(30)}"

    #  Media Ids and Media Types
    media_ids = []
    media_types = []
    num_medias = random.randint(0, 5)
    for _ in range(num_medias):
        media_ids.append(random.randint(1, sys.maxsize))
        media_types.append("PIC")

    return ComposePostServiceParameters(
        req_id=req_id,
        username=username,
        user_id=user_id,
        text=text,
        media_ids=media_ids,
        media_types=media_types,
    )

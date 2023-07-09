"""
This file is used to generate parameters for ComposePostService.
"""
import random
import sys
from dataclasses import dataclass, field
from typing import List

sys.path.append("gen-py")
sys.path.append("utils")

from social_network.ttypes import *
from utility import get_random_string


@dataclass
class ComposePostServiceParameters:
    req_id: int = field(default_factory=lambda: random.getrandbits(63))
    username: str = field(default_factory=lambda: f"username_{random.randint(1, 962)}")
    user_id: int = field(default_factory=lambda: random.randint(1, 962))
    text: str = field(default_factory=lambda: get_random_string(100))
    media_ids: List[int] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    post_type: int = 0

    def __post_init__(self):
        # Add user mentions
        num_user_mentions = random.randint(0, 3)
        user_mention_ids = set()
        while len(user_mention_ids) < num_user_mentions:
            user_mention_id = random.randint(1, 962)
            if user_mention_id != self.user_id:
                user_mention_ids.add(user_mention_id)
        for user_mention_id in user_mention_ids:
            self.text += f" @username_{user_mention_id}"

        # Add URLs
        num_urls = random.randint(0, 3)
        for _ in range(num_urls):
            self.text += f" http://{get_random_string(30)}"

        # Add media
        num_medias = random.randint(0, 5)
        for _ in range(num_medias):
            self.media_ids.append(random.randint(1, sys.maxsize))
            self.media_types.append("PIC")

    def generate_post_class_input(self, req_id=None):
        """
        Generate the input for the Post class.
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


def generate_input_for_compose_post_service(req_id=None):
    """Generate input for ComposePostService"""
    return ComposePostServiceParameters(req_id=req_id)

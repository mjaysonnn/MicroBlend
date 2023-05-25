"""
Insert 1000 posts to mongodb
"""
import random
import sys

sys.path.append("utils")

# Mongo
import pymongo
from pymongo import MongoClient

# Logging to file
logger = None

# Configuration from utils/conf.ini
conf_dict = {}

url_shorten_client = None
social_graph_client = None
post_storage_client = None
user_timeline_client = None
home_timeline_client = None


def init_mongodbs(drop_all_dbs=True, except_social_graph=False):
    global logger
    global conf_dict
    global url_shorten_client
    global social_graph_client
    global post_storage_client
    global user_timeline_client
    global home_timeline_client

    url_shorten_client = MongoClient("url-shorten-mongodb", 27017)
    social_graph_client = MongoClient("social-graph-mongodb", 27017)
    post_storage_client = MongoClient("post-storage-mongodb", 27017)
    user_timeline_client = MongoClient("user-timeline-mongodb", 27017)
    home_timeline_client = MongoClient("home-timeline-mongodb", 27017)

    if drop_all_dbs:
        url_shorten_client.drop_database("url_shorten")
        if not except_social_graph:
            social_graph_client.drop_database("social_graph")
        social_graph_client.drop_database("user")
        post_storage_client.drop_database("post")
        user_timeline_client.drop_database("user_timeline")
        home_timeline_client.drop_database("home_timeline")


def init_social_graph():
    global logger
    global conf_dict
    global social_graph_client

    port = 27017
    if social_graph_client is None:
        social_graph_client = MongoClient(
            "social-graph-mongodb", port, waitQueueTimeoutMS=10000
        )

    # Make index for user db
    user_db = social_graph_client["user"]
    user_collection = user_db["user"]
    user_collection.delete_many({})
    user_collection.create_index(
        [("user_id", pymongo.ASCENDING)], name="user_id", unique=True
    )

    # Make Index for social graph db
    social_graph_db = social_graph_client["social_graph"]
    social_graph_collection = social_graph_db["social_graph"]
    social_graph_collection.delete_many({})
    social_graph_collection.create_index(
        [("user_id", pymongo.ASCENDING)], name="user_id", unique=True
    )

    def register(first_name, last_name, username, password, user_id=None):
        if user_id is None:
            user_id = random.getrandbits(64)
        document = {
            "first_name": first_name,
            "last_name": last_name,
            "username": username,
            "password": password,
            "user_id": user_id,
        }
        user_collection.insert_one(document)

    def follow(user_id, followee_id):
        social_graph_collection.find_one_and_update(
            filter={"user_id": user_id},
            update={"$push": {"followees": followee_id}},
            upsert=True,
        )

    # Register User
    for user_id in range(1, 1000):
        register(
            first_name=f"first_name_{str(user_id)}",
            last_name=f"last_name_{str(user_id)}",
            username=f"username_{str(user_id)}",
            password=f"password_{str(user_id)}",
            user_id=user_id,
        )
    # Follow
    follow(4, 1)
    follow(5, 1)

    print("Done making social graph!")


if __name__ == "__main__":
    print("Starting InitSocialGraph Container")
    init_mongodbs()
    init_social_graph()
    print("Successfully exiting InitSocialGraph Container")

from pathlib import Path
from typing import Union

from fastapi import FastAPI

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

product_db = {
    "products": [
        {
            "id": "OLJCESPC7Z",
            "name": "Sunglasses",
            "description": "Add a modern touch to your outfits with these sleek aviator sunglasses.",
            "picture": "/static/img/products/sunglasses.jpg",
            "priceUsd": 19,
            "categories": ["accessories"],
        },
        {
            "id": "66VCHSJNUP",
            "name": "Tank Top",
            "description": "Perfectly cropped cotton tank, with a scooped neckline.",
            "picture": "/static/img/products/tank-top.jpg",
            "priceUsd": 18,
            "categories": ["clothing", "tops"],
        },
        {
            "id": "1YMWWN1N4O",
            "name": "Watch",
            "description": "This gold-tone stainless steel watch will work with most of your outfits.",
            "picture": "/static/img/products/watch.jpg",
            "priceUsd": 109,
            "categories": ["accessories"],
        },
        {
            "id": "L9ECAV7KIM",
            "name": "Loafers",
            "description": "A neat addition to your summer wardrobe.",
            "picture": "/static/img/products/loafers.jpg",
            "priceUsd": 89,
            "categories": ["footwear"],
        },
        {
            "id": "2ZYFJ3GM2N",
            "name": "Hairdryer",
            "description": "This lightweight hairdryer has 3 heat and speed settings. It's perfect for travel.",
            "picture": "/static/img/products/hairdryer.jpg",
            "priceUsd": 24,
            "categories": ["hair", "beauty"],
        },
        {
            "id": "0PUK6V6EV0",
            "name": "Candle Holder",
            "description": "This small but intricate candle holder is an excellent gift.",
            "picture": "/static/img/products/candle-holder.jpg",
            "priceUsd": 18,
            "categories": ["decor", "home"],
        },
        {
            "id": "LS4PSXUNUM",
            "name": "Salt & Pepper Shakers",
            "description": "Add some flavor to your kitchen.",
            "picture": "/static/img/products/salt-and-pepper-shakers.jpg",
            "priceUsd": 18,
            "categories": ["kitchen"],
        },
        {
            "id": "9SIQT8TOJO",
            "name": "Bamboo Glass Jar",
            "description": "This bamboo glass jar can hold 57 oz (1.7 l) and is perfect for any kitchen.",
            "picture": "/static/img/products/bamboo-glass-jar.jpg",
            "priceUsd": 5,
            "categories": ["kitchen"],
        },
        {
            "id": "6E92ZMYYFZ",
            "name": "Mug",
            "description": "A simple mug with a mustard interior.",
            "picture": "/static/img/products/mug.jpg",
            "priceUsd": 8,
            "categories": ["kitchen"],
        },
    ]
}


def get_cart_price(cart):
    """
    Get the total price of the cart
    """
    total_price = 0
    for product, q in cart.items():
        for product_items in product_db["products"]:
            if product_items["id"] == product:
                total_price += int(product_items["priceUsd"]) * int(q)
    return total_price


@app.get("/product_catalog_service/{input_p}")
def run_product_catalog(input_p: Union[str, None] = None):
    """
    Invoke get_cart_price
    """
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    cart = parsed_inputs.get("cart", {"OLJCESPC7Z": 1})

    return get_cart_price(cart)

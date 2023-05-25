import dataclasses
import random
import string
import time
from pathlib import Path
from typing import Union

import requests
from fastapi import FastAPI

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

product_id_list = [
    "OLJCESPC7Z",
    "66VCHSJNUP",
    "1YMWWN1N4O",
    "L9ECAV7KIM",
    "2ZYFJ3GM2N",
    "0PUK6V6EV0",
    "LS4PSXUNUM",
    "9SIQT8TOJO",
    "6E92ZMYYFZ",
]


@dataclasses.dataclass
class Review:
    review_id: int
    user_id: int
    req_id: int
    text: str
    movie_id: str
    rating: int
    timestamp: int


currency_dict = {
    "EUR": "1.0",
    "USD": "1.1305",
    "JPY": "126.40",
    "BGN": "1.9558",
    "CZK": "25.592",
    "DKK": "7.4609",
    "GBP": "0.85970",
    "HUF": "315.51",
    "PLN": "4.2996",
    "RON": "4.7463",
    "SEK": "10.5375",
    "CHF": "1.1360",
    "ISK": "136.80",
    "NOK": "9.8040",
    "HRK": "7.4210",
    "RUB": "74.4208",
    "TRY": "6.1247",
    "AUD": "1.6072",
    "BRL": "4.2682",
    "CAD": "1.5128",
    "CNY": "7.5857",
    "HKD": "8.8743",
    "IDR": "15999.40",
    "ILS": "4.0875",
    "INR": "79.4320",
    "KRW": "1275.05",
    "MXN": "21.7999",
    "MYR": "4.6289",
    "NZD": "1.6679",
    "PHP": "59.083",
    "SGD": "1.5349",
    "THB": "36.012",
    "ZAR": "16.0583",
}
currency_list = list(currency_dict.keys())


@dataclasses.dataclass
class Address:
    street: str
    city: str
    state: str
    zipcode: int


@dataclasses.dataclass
class CreditCardInfo:
    card_number: str
    card_month: int
    card_year: int


@dataclasses.dataclass
class OrderResult:
    order_id: str
    shipping_tracking_id: str
    shipping_address: str
    transaction_id: str
    cart_items: dict


@dataclasses.dataclass
class OrderRequest:
    user_id: int
    user_currency: str
    street: str
    city: str
    state: str
    zip_code: int
    card_number: str
    card_month: int
    card_year: int
    user_email: str


def generate_input_for_place_order():
    # random string
    user_id = random.getrandbits(10)

    # user_currency
    user_currency = random.choice(currency_list)

    street = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    city = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    state = "".join(random.choices(string.ascii_letters + string.digits, k=4))

    # random number of 5
    zip_code = random.randint(10000, 99999)

    card_number = "".join(random.choices(string.digits, k=16))

    card_month = random.randint(1, 12)
    card_year = random.randint(2020, 2025)

    user__email = (
            "".join(random.choices(string.ascii_letters + string.digits, k=4))
            + "@gmail.com"
    )
    return OrderRequest(
        user_id,
        user_currency,
        street,
        city,
        state,
        zip_code,
        card_number,
        card_month,
        card_year,
        user__email,
    )


def invoke_product_catalog_service(cart):
    """
    # 1. Get Text from TextService
    """

    input_d = {"cart": cart}

    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://product-catalog-service:5001/product_catalog_service/{encoded_input}"

    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def invoke_currency_service(total_cost, user_currency):
    """
    # 2. Convert to user set currency
    """

    input_d = {"total_cost": total_cost, "user_currency": user_currency}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://currency-service:5002/currency_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    total_cost = resp.json()

    return total_cost


def invoke_payment_service(credit_card_info, cost_in_user_currency):
    """
    # 3. Make Payment
    """

    input_d = {
        "credit_card_info": credit_card_info,
        "cost_in_user_currency": cost_in_user_currency,
    }
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://payment-service:5003/payment_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()

    return resp.json()


def invoke_shipping_service(address_info):
    """
    # 4. Shipping
    """

    input_d = {"address_info": address_info}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://shipping-service:5004/shipping_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def invoke_email_service(user_email, order_result_instance):
    """
    # 5. Send Email
    """

    input_d = {
        "user__email": user_email,
        "order_result_instance": order_result_instance,
    }
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://email-service:5005/email_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


def PlaceOrder(order_request):
    """
    Place Order
    """

    cart_id = "".join(random.choices(string.ascii_letters + string.digits, k=10))

    number_of_products = random.randint(1, 10)
    product_number_list = list(
        {random.randint(0, 9) for _ in range(number_of_products)}
    )
    product_ids = [product_id_list[i - 1] for i in product_number_list]
    item_quantity = [random.randint(1, 9) for _ in range(number_of_products)]
    cart = dict(zip(product_ids, item_quantity))

    total_cost = invoke_product_catalog_service(cart)

    user_currency = order_request.user_currency
    cost_in_user_currency = invoke_currency_service(total_cost, user_currency)

    card_number = order_request.card_number
    card_month = order_request.card_month
    card_year = order_request.card_year
    credit_card_info = CreditCardInfo(card_number, card_month, card_year)
    transaction_id = invoke_payment_service(credit_card_info, cost_in_user_currency)

    street = order_request.street
    city = order_request.city
    state = order_request.state
    zip_code = order_request.zip_code
    address_info = Address(street, city, state, zip_code)
    shipping_id = invoke_shipping_service(address_info)

    user_email = order_request.user_email
    order_result_instance = OrderResult(
        cart_id, shipping_id, user_email, transaction_id, cart
    )
    invoke_email_service(user_email, order_result_instance)


@app.get("/checkout_service/{input_p}")
def run_checkout_service(input_p: Union[str, None] = None):
    """
    Run the service
    """
    start_time = time.time()

    order_request = generate_input_for_place_order()

    PlaceOrder(order_request)

    end_time = time.time()

    # return the response time in milliseconds
    logger.info((end_time - start_time) * 1000)

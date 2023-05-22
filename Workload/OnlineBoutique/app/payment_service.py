import dataclasses
import random
import string
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# Import OpenTelemetry and Logger modules
from utils import utils

# import pymongo
# from pymongo import MongoClient

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer


# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()

# make dictionary which indicates 1 will be VISA, 2 will be MasterCard, 3 will be Amex
card_type = {1: "VISA", 2: "MasterCard", 3: "Amex"}


@dataclasses.dataclass
class CreditCardInfo:
    card_number: str
    card_month: int
    card_year: int


def make_payment(credit_card_info: CreditCardInfo, amount: float):
    """
        Invoke another service"""

    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("MakePayment", parent_ctx, kind=trace.SpanKind.SERVER):
    # if first number of credit card number is not 1, or 2, or 3, return False

    # logger.debug(f"Credit Card Info: {credit_card_info}")
    # logger.debug(f"Credit Card Info: {credit_card_info.card_number}")
    # logger.debug(f"Credit Card Info: {type(credit_card_info.card_number)}")
    x = int(credit_card_info.card_number[0]) % 3
    if x not in card_type:
        return False

    # validate credit card month
    if credit_card_info.card_month not in range(1, 13):
        return False

    # validate credit card year
    if credit_card_info.card_year not in range(2021, 2031):
        return False

    transaction_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    # Mention how much it will be paid
    # logger.info(f"Payment of {amount} with transaction_id {transaction_id} will be made")

    return transaction_id


@app.get("/payment_service/{input_p}")
def run_make_payment_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Get the input parameters
    credit_card_info = parsed_inputs.get('credit_card_info', CreditCardInfo("1234567890123456", 1, 2021))
    amount = parsed_inputs.get('amount', 100.0)

    # 3. Make the payment
    transaction_id = make_payment(credit_card_info, amount)

    # 4. Return the output
    return transaction_id



def main():
    pass


if __name__ == "__main__":
    main()

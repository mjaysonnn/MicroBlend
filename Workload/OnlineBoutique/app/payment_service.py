import dataclasses
import random
import string
from pathlib import Path
from typing import Union

from fastapi import FastAPI

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# make dictionary which indicates 1 will be VISA, 2 will be MasterCard, 3 will be Amex
card_type = {1: "VISA", 2: "MasterCard", 3: "Amex"}


@dataclasses.dataclass
class CreditCardInfo:
    card_number: str
    card_month: int
    card_year: int


def make_payment(credit_card_info: CreditCardInfo, amount: float):
    """
    Make payment
    """

    x = int(credit_card_info.card_number[0]) % 3
    if x not in card_type:
        return False

    # validate credit card month
    if credit_card_info.card_month not in range(1, 13):
        return False

    # validate credit card year
    if credit_card_info.card_year not in range(2021, 2031):
        return False

    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


@app.get("/payment_service/{input_p}")
def run_make_payment_service(input_p: Union[str, None] = None):
    """
    Run make payment service
    """
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Get the input parameters
    credit_card_info = parsed_inputs.get(
        "credit_card_info", CreditCardInfo("1234567890123456", 1, 2021)
    )
    amount = parsed_inputs.get("amount", 100.0)

    return make_payment(credit_card_info, amount)


def main():
    """
    Main function for solving dependency issue
    """
    pass


if __name__ == "__main__":
    main()

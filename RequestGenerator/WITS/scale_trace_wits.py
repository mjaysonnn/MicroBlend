"""
Divide Workload to 20 minutes with scaling factor

SYNOPSIS
========
::
    python3 scale_trace.py

DESCRIPTION
==========
1. Open workload
2. Make it 60 minutes
3. Scale by factor
4. Check average or median requests
5. Save it to csv

ENVIRONMENT
===========
1. File to read
2. New file to save
"""

# Library imports
import os
import sys
from inspect import getframeinfo, currentframe

import pandas as pd
import logging

# Logging Configuration =====
modules_for_removing_debug = [
    "urllib3",
    "s3transfer",
    "boto3",
    "botocore",
    "urllib3",
    "requests",
    "paramiko",
]
for name in modules_for_removing_debug:
    logging.getLogger(name).setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-6s] [%(filename)s:%(lineno)-4s]  %(message)s"
)
# Console Handler =====
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# ===== SET LEVEL =====
logger.setLevel(logging.DEBUG)

# ===== Workload configuration =====
WORKLOAD_NAME = "wits.csv"
WORKLOAD_TO_READ = os.path.join("/WITS", WORKLOAD_NAME)
WORKLOAD_TO_READ = "wits.csv"
logger.info(os.getcwd())
logger.info(f"File to Read : {WORKLOAD_TO_READ}")
WORKLOAD_NAME_WITHOUT_EXTENSION = WORKLOAD_NAME.split(".")[0]

# ===== Configuration =====
# Duration =====
MINUTES_DURATION = 60  # TODO:Change
SECONDS_DURATION = 60 * MINUTES_DURATION  # TODO:Change
# Scaling Factor =====
SCALING_FACTOR = 6.05  # TODO:Change


def main():
    # Read CSV =====
    df = pd.read_csv(WORKLOAD_TO_READ, usecols=[1], names=["Req"])

    # logger.info(df)
    # sys.exit(getframeinfo(currentframe()))

    # ===== ORIGINAL INFO =====
    logger.info("Original Info")
    logger.info(f"Describe : \n{df.describe()}")
    logger.info(f"Shape : {df.shape}\n")

    # ===== Divide by SCALING_FACTOR =====
    logger.info(f"Divide by {SCALING_FACTOR}")
    df["Req"] = df["Req"].div(SCALING_FACTOR).astype(int)
    df_scaled = df.astype(int)
    logger.info(f"Describe : \n{df_scaled.describe()}")
    logger.info(f"Shape : {df_scaled.shape}\n")
    logger.info(f"Average : {df_scaled.mean()}")
    average_req_for_division = int(df_scaled['Req'].mean())
    # logger.info(average_req)

    duration_to_configure = int(SECONDS_DURATION / 60)
    logger.info(f"Cut to {duration_to_configure} minutes")
    df_configured_duration = df_scaled.iloc[0:SECONDS_DURATION]
    df_configured_duration.index = df_configured_duration.index + 1
    logger.info(f"Describe : \n{df_configured_duration.describe()}")
    logger.info(f"Shape : {df_configured_duration.shape}\n")
    logger.info(f"Average for : {df_configured_duration.mean()}")
    average_req_for_60 = int(df_configured_duration['Req'].mean())

    logger.info("Cut to 30 minutes")
    df_30_mins = df_configured_duration.iloc[0:1800]
    # df_30_mins.index = df_30_mins.index + 1
    logger.info(f"Describe : \n{df_30_mins.describe()}")
    logger.info(f"Shape : {df_30_mins.shape}\n")

    logger.info("Cut to 30 minutes")
    df_20_mins = df_configured_duration.iloc[0:1200]
    # df_30_mins.index = df_30_mins.index + 1
    logger.info(f"Describe : \n{df_20_mins.describe()}")
    logger.info(f"Shape : {df_20_mins.shape}\n")

    logger.info("Cut to 10 minutes")
    df_10_mins = df_configured_duration.iloc[0:600]
    # df_10_mins.index = df_10_mins.index + 1
    logger.info(f"Describe : \n{df_10_mins.describe()}")
    logger.info(f"Shape : {df_10_mins.shape}\n")

    logger.info("Cut to 5 minutes")
    df_5_minutes = df_configured_duration.iloc[0:300]
    # df_5_minutes.index = df_5_minutes.index + 1
    logger.info(f"Describe : \n{df_5_minutes.describe()}")
    logger.info(f"Shape : {df_5_minutes.shape}\n")

    logger.info("Cut to 1 minutes")
    df_1_minutes = df_5_minutes[:60]
    logger.info(f"Describe : \n{df_1_minutes.describe()}")
    logger.info(f"Shape : {df_1_minutes.shape}\n")

    # Total request per 1 minute =====
    logger.info("Check for every 1 minute")
    for i in range(0, SECONDS_DURATION, 60):
        df_temp = df_configured_duration[i: i + 60]
        logger.info(df_temp.values.sum())
    logger.debug(df_temp[df_temp > 13].count())

    # Save it to 60 minute csv =====
    logger.info(f"Save it to {int(SECONDS_DURATION / 60)} minute csv =====")
    duration_for_workload = int(SECONDS_DURATION / 60)
    file_name = (
        f"{WORKLOAD_NAME_WITHOUT_EXTENSION}_"
        f"average_{average_req_for_60}_factor_{int(SCALING_FACTOR)}_minute_{duration_for_workload}.csv"
    )
    logger.info(f"File Name is {file_name}")
    df_configured_duration.to_csv(file_name, header=False, index=False)

    # Save it to 30 minute csv =====
    file_name = (
        f"{WORKLOAD_NAME_WITHOUT_EXTENSION}_"
        f"average_{average_req_for_60}_factor_{int(SCALING_FACTOR)}_minute_30.csv"
    )
    logger.info(f"File Name is {file_name}")
    df_30_mins.to_csv(file_name, header=False, index=False)

    # Save it to 20 minute csv =====
    file_name = (
        f"{WORKLOAD_NAME_WITHOUT_EXTENSION}_"
        f"average_{average_req_for_60}_factor_{int(SCALING_FACTOR)}_minute_20.csv"
    )
    logger.info(f"File Name is {file_name}")
    df_20_mins.to_csv(file_name, header=False, index=False)

    # Save it to 10 minute csv =====
    file_name = (
        f"{WORKLOAD_NAME_WITHOUT_EXTENSION}_"
        f"average_{average_req_for_60}_factor_{int(SCALING_FACTOR)}_minute_10.csv"
    )
    logger.info(f"File Name is {file_name}")
    df_10_mins.to_csv(file_name, header=False, index=False)

    # Save it to 5 minute csv =====
    file_name = (
        f"{WORKLOAD_NAME_WITHOUT_EXTENSION}_"
        f"average_{average_req_for_60}_factor_{int(SCALING_FACTOR)}_minute_5.csv"
    )
    logger.info(f"File Name is {file_name}")
    df_5_minutes.to_csv(file_name, header=False, index=False)

    # Save it to 1 minute csv =====
    file_name = (
        f"{WORKLOAD_NAME_WITHOUT_EXTENSION}_"
        f"average_{average_req_for_60}_factor_{int(SCALING_FACTOR)}_minute_1.csv"
    )
    logger.info(f"File Name is {file_name}")
    df_1_minutes.to_csv(file_name, header=False, index=False)


if __name__ == "__main__":
    main()

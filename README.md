# MicroBlend

MicroBlend is a framework designed to automatically integrate IaaS and FaaS services, focusing on both cost efficiency and performance optimization.

## Components

- **Compiler**: Transforms code to be FaaS-compatible.
- **Controller**: Manages autoscaling and allocates services.
- **Traces**: Utilizes real-world traces (WITS, WIKI) for evaluation purposes.
- **Workload**: A benchmark tool used for evaluation, written in Python.

## Prerequisites

- **Loadcat**: Essential for controlling nginx, which acts as a load balancer. You can access the [Loadcat Repository here](https://github.com/mjaysonnn/loadcat.git).

## Setup and Usage

1. **Install Loadcat**: Once installed, initialize several servers and add them to Loadcat.
2. **Deploy the Workload**: Ensure the workload is appropriately distributed across the servers. (Require background on Docker)
3. **Run the Controller**: Further details can be found in `Controller/controller.py`.

---
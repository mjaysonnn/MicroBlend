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
* Once the workload is deployed, the one of microservices will  initiate Prometheus, which will be responsible for collecting `runq_latency` metrics for each microservice. This data can be subsequently used to feed into a training model, enabling the selection of microservices that are best suited to meet the Service Level Objectives (SLO). During the provisioning phase, a compiler is utilized to transform the microservice into a Lambda function. Additionally, compiler modifies the orchestrator function, transitioning its calls from a VM-based function to Lambda. And controller.py would reroute requests to run hybrid code through Loadcat.

3. **Run the Controller**: Further details can be found in `Controller/controller.py`.

---

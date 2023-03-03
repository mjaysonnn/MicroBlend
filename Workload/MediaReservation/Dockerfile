# syntax=docker/dockerfile:1
# Use the ebpf exporter image as a parent image.
FROM smj8612/ebpf_exporter:latest

# install packages without prompting the user for choices
ARG DEBIAN_FRONTEND=noninteractive

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Install packages for python
RUN apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN apt-get install -y curl
RUN apt-get -y install python3.8-venv

# Virtual environment
RUN python3.8 -m venv /venv

# Upgrade pip
RUN /venv/bin/python3.8 -m pip install --upgrade pip

# PATH is an environment variable that tells the shell where to look for executables
ENV PATH="/venv/bin:$PATH"

# Move all files to source directory
WORKDIR /code

# Copy the requirements.txt file to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install the requirements
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the files to the working directory
COPY ./app /code/app
#COPY ./app/gen-py /gen-py

# Add Libraries
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

# Working Directory for executing commands
WORKDIR /code/app
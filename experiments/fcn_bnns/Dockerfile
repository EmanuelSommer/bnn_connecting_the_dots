# Use an official lightweight Python base image
FROM python:3.10-slim-buster

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Install git
RUN apt-get update && apt-get install -y git

# Argument for GitHub Token
ARG GITHUB_TOKEN

# Create .netrc file for GitHub authentication
RUN echo "machine github.com login $GITHUB_TOKEN password x-oauth-basic" > ~/.netrc

# Copy the dependencies file to the working directory
COPY experiments/fcn_bnns/requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY data/ data/
COPY experiments/ experiments/
COPY module_sandbox/ module_sandbox/
COPY results/de/ results/de/

# Command to run on container start
CMD [ "python", "experiments/fcn_bnns/fcn_bnns_main.py" ]

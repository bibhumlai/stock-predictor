#!/usr/bin/env bash

# This script is a basic example of how to discover GPUs on a node.
# It uses nvidia-smi to find the indices of the GPUs.
# It outputs a JSON string that Spark can use to schedule tasks on GPUs.

ADDRS=$(nvidia-smi --query-gpu=index --format=csv,noheader | sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/","/g')
echo "{\"name\": \"gpu\", \"addresses\": [\"$ADDRS\"]}"

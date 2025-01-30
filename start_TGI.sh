#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path-to-volume> <model-id>"
    exit 1
fi

# Assign command-line arguments to variables
VOLUME=$(realpath "$1")
MODEL_ID=$2

# Run the Docker command with the provided arguments
docker run --gpus all --shm-size 1g -p 8080:80 -v "$VOLUME:/data" \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id "$MODEL_ID" # --quantize bitsandbytes
#!/bin/bash

# Set model path variable
model_path=${2:-$(pwd)/checkpoints/resnet_50/final.pt}

echo Running inference on data in $1 with model at $model_path
python3 -m aim_perception.run_inference -d $1 -m $model_path

echo Inference completed.
echo Results at $1/results.csv

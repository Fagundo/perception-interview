#!/bin/bash

# Solve arguments
training_data_dir=$1
inference_data_dir=$2
model_path=checkpoints/$(date +%Y-%m-%d_%T)/model.pt

# Install environment
echo Installing python environment...
echo ---------------------------
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m venv ./venv
source ./venv/bin/activate
echo

echo Installing requirements.txt
echo ---------------------------
python3 -m pip install --no-input -r requirements.txt
echo 

# Run pytests
echo Running integration tests...
echo ---------------------------
python3 -m pytest tests/
echo Good to go!

# Run training
echo Running training on data in $1 for 55 epochs. 
echo Model will be saved to $model_path
echo ---------------------------
mkdir $model_path
python3 -m aim_perception.run_training -d $1 -m $model_path -e 1 -b 100


# Run training
echo Running inference data in $2...
echo Model will be saved to $model_path
echo ---------------------------
python3 -m aim_perception.run_inference -d $2
echo Inference completed.
echo Results at $2/results.csv
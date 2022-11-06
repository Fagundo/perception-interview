#!/bin/bash

# Solve arguments
training_data_dir=$1
model_path=checkpoints/$(date +%Y-%m-%d_%T)

# Install environment
echo Installing python environment...
echo ---------------------------
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m venv ./venv
echo

echo Installing requirements.txt
echo ---------------------------
source ./venv/bin/activate
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
python3 -m aim_perception.run_training -d $1 -m $model_path/model.pt
echo Training and evaluation completed, model saved to $model_path/model.pt
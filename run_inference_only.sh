#!/bin/bash

echo Running inference on data in $1
python3 -m aim_perception.run_inference -d $1

echo Inference completed.
echo Results at $1/results.csv

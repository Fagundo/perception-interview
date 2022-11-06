#!/bin/bash
epochs=${3:-45}

echo Running training on data in $1 for $epochs epochs and saving model to $2
python3 -m aim_perception.run_training -d $1 -m $2 -e $epochs

echo Model saved to $2
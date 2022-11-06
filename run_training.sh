#!/bin/bash
epochs=${3:-55}
batch_size=${4:-256}

echo Running training on data in $1 for $epochs epochs and saving model to $2
python3 -m aim_perception.run_training -d $1 -m $2 -e $epochs -b $batch_size

echo Model saved to $2
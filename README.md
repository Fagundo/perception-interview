# AIM Perception
Code repository for AIM perception code interview.

# Instructions

## Run Training
Running `train.sh` will execute the following steps...
1. Create a python3 virtual env and install necessary packages.
2. Run integration tests with pytest. This tests training and inference execution.
3. Run training with preset parameters on provided dataset (you must provide the datset path).  
 - The model is saved to the checkpoints directory.
 - The training dataset must be as outlined in project instructions. Ie, it contains a ground_truth.csv and the `data` directory with respective image files.

To run, make sure you have python3 installed, specifically python3.8 or python3.9
`source train.sh /path/to/training/dataset`

## Run Inference Only
With the dependencies in requirements.txt installed, you can run inference on the pretrained model in `checkpoints/resnet_50/final.pt` or on a model train with the steps above. 

1. To run on pretrained model: `sh inference.sh /path/to/inference/data`
2. To run on a different model: `sh inference.sh /path/to/inference/data /path/to/model.pt`

## Sweep Models
Sweeping the models enables you to try out various ResNet sizes with different hyperparameters. Feel free to give it a go!
1. Configure Weights and Biases (WandB)
    - Create an account at https://wandb.ai/
    - With WandB installed, run `wandb login` in the terminal

2. Run a sweep with the following command `python -m sweep -p <path/to/data> -w wandb_project_name ...`
    - Please look at sweep.py for sweep parameters
    - Example `python -m sweep -p /home/ubuntu/aim/vehicle_dataset -w resnet_sweep -d 0.05 0.0 -b 256 -i 128 -wd 0.00001 -r 50 18 34`

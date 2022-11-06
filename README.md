# AIM Perception
Code repository for AIM perception code interview.

# Instructions

## Run End to End
Running `run_end_to_end.sh` will execute the following steps...
1. Create a python3 virtual env and install necessary packages.
2. Run integration tests with pytest. This tests training and inference execution.
3. Run training with preset parameters on provided dataset (you must provide the datset path).  
 - The model is saved to the checkpoints directory.
 - The training dataset must be as outlined in project instructions. Ie, it contains a ground_truth.csv and the `data` directory with respective image files.
4. Runs inference using the model that just trained on a directory of images you provide.
 - Output is saved as 'results.csv' in the directory of data you provide for inference

To run, make sure you have python3 installed, specifically python3.8 or python3.9
`source run_end_to_end.sh /path/to/training/dataset /path/to/inference/data`

## Run Training Only
If you want to skip the entire pipeline and have the dependencies in requirements.txt installed, you can run training only via the `run_training_only.sh`...

To execute, run `sh run_training_only.sh /path/to/training/dataset /path/to/save/checkpoint/model.pt <optional: epochs, default: 55> <optional: batch_size, default: 256>`

## Run Inference Only
If you want to skip the entire pipeline and have the dependencies in requirements.txt installed, you can run inference on the model I trained via the `run_inference_only.sh`...

To execute, run `sh run_inference_only.sh /path/to/inference/data`

## Sweep Models
Sweeping the models enables you to try out various ResNet sizes with different hyperparameters. Feel free to give it a go!
1. Configure Weights and Biases (WandB)
    - Create an account at https://wandb.ai/
    - With WandB installed, run `wandb login` in the terminal

2. Run a sweep with the following command `python -m sweep -p <path/to/data> -w wandb_project_name ...`
    - Please look at sweep.py for sweep parameters
    - Example `python -m sweep -p /home/ubuntu/aim/vehicle_dataset -w resnet_sweep -d 0.05 0.0 -b 256 -i 128 -wd 0.00001 -r 50 18 34`

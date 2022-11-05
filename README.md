# AIM Perception
Code repository for AIM perception code interview.

# Instructions
Foremost, install the packages required in `requirements.txt`

```
conda create -n aim python=3.8
pip install -r requirements.txt
conda activate aim
```

## Sweep Models
1. Configure Weights and Biases (WandB)
    - Create an account at https://wandb.ai/
    - With WandB installed, run `wandb login` in the terminal

2. Run a sweep with the following command `python -m sweep -p <path/to/data> -w wandb_project_name ...`
    - Please look at sweep.py for sweep parameters
    - Example `python -m sweep -p /home/ubuntu/aim/vehicle_dataset -w resnet_sweep -d 0.05 0.0 -b 256 -i 128 -wd 0.00001 -r 50 18 34`

## Run Final Model 
### No Retrain (Test Eval Only)
`python -m run_final -d <path/to/data> -m <path/to/this/repository/checkpoints/resnet_50/model.pt>`

### Run Final Model with Retrain
`python -m run_final --retrain -d <path/to/data> -m <path/to/this/repository/checkpoints/resnet_50/model.pt>`
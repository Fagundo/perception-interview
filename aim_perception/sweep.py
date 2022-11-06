import argparse
import itertools
from typing import List
from aim_perception import pipeline

'''
Script for sweeping over models and parameters to identify optimal configuration of ResNet size and params.

Allows sweeping for key parameters:
    - Model type (ResNet18, ResNet34 and ResNet50)
    - Dropout probabilities for final fully connected layer
    - Batch sizes for training
    - Image sizes for image resize transformation (given the large distribution in image sizes)
    - Weight decay for regularization

Results are reported to WandB, which allows us to leverage parallel coordinate plots to look at what
params help model performance.

Please, only run this if you have a GPU and WandB account. Otherwise, it will take a very long time and you wont
get all the pretty plots.

WandB: https://wandb.ai/
'''

def run_sweep(
    root_data_path: str,
    fine_tune_epochs: int,
    resnet_sizes: List[int],
    dropouts: List[float],
    batch_sizes: List[int], 
    image_sizes: List[int], 
    weight_decays: List[int], 
    wandb_project: str,
) -> None:

    ''' Function to run a sweep of model training runs over model types and parameters. 

    Args:
        root_data_dir (str): Root directory for data, of structure:
            /
            - data/
                - <image_name>.jpg
            - ground_truth.csv
        fine_tune_epochs (int): Number of epochs to freeze ResNet backbone for. 
            Suggest doing it for the same value as the LR scheduler step size (15). 
            If you unfreeze at a high learning rate, the model's loss will jump.
        resnet_sizes (List[int]): Sizes of resnet to sweep over. Supported 18, 34, and 50. 
        dropouts (List[float]): Dropout values to sweep over
        batch_sizes (List[int]): Batch sizes to sweep over
        image_sizes (List[int]): Image size to sweep over. We make square images without augementing aspect ratio.
        weight_decays (List[int]): Weight decay values to sweep over
        wandb_project (str): WandB project to report to
    '''

    # Enforce types
    resnet_sizes = list(map(lambda x: int(x), resnet_sizes))    
    dropouts = list(map(lambda x: float(x), dropouts))    
    batch_sizes = list(map(lambda x: int(x), batch_sizes))    
    image_sizes = list(map(lambda x: int(x), image_sizes))    
    weight_decays = list(map(lambda x: float(x), weight_decays))    

    # Double check resnet sizes:
    for resnet_size in resnet_sizes:
        assert resnet_size in [18, 34, 50], f'Resnet size {resnet_size} not supported!'

    # Iterate over data loaders
    for image_size, batch_size in itertools.product(image_sizes, batch_sizes):

        # Create loaders
        train_loader, val_loader, _ = pipeline.create_data_loaders(
            root_data_path=root_data_path, image_size=(image_size, image_size), batch_size=batch_size
        )

        # Iterate over model params
        for resnet_size, weight_decay, dropout in itertools.product(resnet_sizes, weight_decays, dropouts):

            # Instantiate model and optimizer
            model_name = f'resnet_{resnet_size}'
            torch_model, optimizer = pipeline.create_model_and_optimimzer(
                model_name, dict(dropout=dropout), weight_decay=weight_decay
            )

            # Configure wandb
            wandb_config = dict(resnet_size=resnet_size, weight_decay=weight_decay, image_size=image_size, dropout=dropout)

            # Run training
            pipeline.train_model(
                torch_model, 
                optimizer, 
                train_loader, 
                val_loader, 
                wandb_project=wandb_project, 
                wandb_run_name=model_name,
                wandb_config_updates=wandb_config,
                fine_tune_epochs=fine_tune_epochs
            )
      
if __name__=='__main__':

    parser = argparse.ArgumentParser(prog = 'Model Sweeper')
    parser.add_argument('-p','--path', help='Data path', required=True)
    parser.add_argument('-w','--wandb', help='Wandb project name', required=True)
    parser.add_argument('-f','--fine_tune_epochs', help='Number of epochs to freeze backbone', default = 15)
    parser.add_argument('-r','--resnet_sizes', nargs='+', help='Resnet sizes to iterate over', default=[18, 34, 50])
    parser.add_argument('-d','--dropout', nargs='+', help='Model dropout values', default=[0.1, 0.05])
    parser.add_argument('-b','--batch_sizes', nargs='+', help='Training batch sizes', default=[256])
    parser.add_argument('-i','--image_sizes', nargs='+', help='Image sizes', default=[96])    
    parser.add_argument('-wd','--weight_decays', nargs='+', help='Weight decays', default=[1e-4]) 

    args = parser.parse_args()

    run_sweep(
        root_data_path=args.path,
        resnet_sizes=args.resnet_sizes,
        dropouts=args.dropout,
        batch_sizes=args.batch_sizes,
        image_sizes=args.image_sizes,
        weight_decays=args.weight_decays,
        wandb_project=args.wandb,
        fine_tune_epochs=args.fine_tune_epochs
    )
import argparse
import itertools
from torch import nn
from typing import List
from aim_perception import pipeline


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
            
            # Enforce integers
            dropout = int(dropout)
            resnet_size = int(resnet_size)
            weight_decay = int(weight_decay)

            # Instantiate model and optimizer
            model_name = f'resnet_{resnet_size}'
            torch_model, optimizer = pipeline.create_model_and_optimimzer(
                model_name, dict(dropout=dropout), weight_decay=weight_decay
            )

            # Configure wandb
            wandb_config = dict(resnet_size=resnet_size, weight_decay=weight_decay, image_size=image_size)

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
    parser.add_argument('-f','--fine_tune_epochs', help='Number of epochs to freeze backbone', default = 10)
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
import argparse
from torch import nn
from typing import List
from aim_perception import pipeline


def create_models(dropouts: List[float]) -> List[nn.Module]:
    # Create some models, iterating over dropout, and resnet size
    models = []
    
    for dropout in dropouts:
        for resnet_size in [18, 34]:
            models.extend([
                dict(name=f'resnet_{resnet_size}_imagenet', kwargs=dict(dropout=dropout)),
                dict(name=f'resnet_{resnet_size}', kwargs=dict(dropout=dropout, depthwise=True)),
                dict(name=f'resnet_{resnet_size}', kwargs=dict(dropout=dropout, depthwise=False)),
            ])

    return models


def run_sweep(
    root_data_path: str, 
    dropouts: List[float],
    batch_sizes: List[int], 
    image_sizes: List[int], 
    weight_decays: List[int], 
    fine_tune_epochs: int,
    wandb_project: str,
) -> None:

    # Generate models
    models = create_models(dropouts)

    # Iterate over data loaders
    for image_size in image_sizes:
        image_size = (int(image_size), int(image_size))        

        for batch_size in batch_sizes:
            batch_size = int(batch_size)
            
            # Create loaders
            train_loader, val_loader, _ = pipeline.create_data_loaders(
                root_data_path=root_data_path, image_size=image_size, batch_size=batch_size
            )

            # Iterate over model configs
            for model in models:
                for weight_decay in weight_decays:
                    weight_decay = int(weight_decay)

                    # Instantiate model and optimizer
                    torch_model, optimizer = pipeline.create_model_and_optimimzer(
                        model['name'], model['kwargs'], weight_decay=weight_decay
                    )

                    # Decide on fine tune
                    if 'image_net' in model['name']:
                        fine_tune_temp = fine_tune_epochs
                    else:
                        fine_tune_temp = None

                    pipeline.train_model(
                        torch_model, 
                        optimizer, 
                        train_loader, 
                        val_loader, 
                        wandb_project=wandb_project, 
                        fine_tune_epochs=fine_tune_temp
                    )
      
if __name__=='__main__':

    parser = argparse.ArgumentParser(prog = 'Model Sweeper')
    parser.add_argument('-p','--path', help='Data path', required=True)
    parser.add_argument('-w','--wandb', help='Wandb project name', default=None)
    parser.add_argument('-f','--fine_tune_epochs', help='Number of epochs to freeze backbone', default = 4)
    parser.add_argument('-d','--dropout', nargs='+', help='Model dropout values', default=[0.1, 0.5])
    parser.add_argument('-b','--batch_sizes', nargs='+', help='Training batch sizes', default=[128, 256])
    parser.add_argument('-i','--image_sizes', nargs='+', help='Image sizes', default=[64, 96])    
    parser.add_argument('-wd','--weight_decays', nargs='+', help='Image sizes', default=[1e-4, 1e-5])    

    args = parser.parse_args()

    run_sweep(
        root_data_path=args.path,
        dropouts=args.dropout,
        batch_sizes=args.batch_sizes,
        image_sizes=args.image_sizes,
        weight_decays=args.weight_decays,
        fine_tune_epochs=int(args.fine_tune_epochs),
        wandb_project=args.wandb,
    )
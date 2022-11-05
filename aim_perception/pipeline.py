import torch
import torchcontrib
from torch import nn
from typing import Tuple, List
from torchvision import transforms
from aim_perception import ModelFactory
from aim_perception.training import Trainer
from aim_perception.loading import AimDatasetConstructor

# Configure Paths
IMAGE_PATH = 'data'
LABEL_PATH = 'ground_truth.csv'

# Configure Normalization. From EDA notebook
IMAGE_MEAN = [0.4876, 0.4846, 0.4841]
IMAGE_STD = [0.2448, 0.2435, 0.2481]


def create_data_loaders(root_data_path: str, image_size: Tuple[int], batch_size: int) -> Tuple[torch.utils.data.DataLoader]:
    ''' Function to create data loaders for train, val and test splits.

    Args:
        root_data_path (str): Root directory for data, of structure:
            /
            - data/
                - <image_name>.jpg
            - ground_truth.csv
        image_size (Tuple[int]): Size to convert images to
        batch_size (int): Batch size for training. Note, validation and test splits will be double this.

    Returns:
        train_loader (torch.utils.data.DataLoader): Data loader for AimDataset of train split
        val_loader (torch.utils.data.DataLoader): Data loader for AimDataset of val split
        test_loader (torch.utils.data.DataLoader): Data loader for AimDataset of test split

    '''
    # Create Data
    dataset_constructor = AimDatasetConstructor(
        root_dir=root_data_path,
        csv_path=LABEL_PATH,
        data_subdir=IMAGE_PATH,
        transforms=[
            transforms.ToTensor(),
            transforms.Resize(size=image_size),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ]
    )

    train, val, test = dataset_constructor.get_all_datasets()

    # Create loaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size*2, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size*2, shuffle=True, num_workers=2)

    return train_loader, val_loader, test_loader

def create_model_and_optimimzer(
    model_name: str, 
    model_kwargs: dict, 
    learning_rate: float = 1e-1, 
    weight_decay: float = 1e-4, 
    momentum: float = 0.9,
    swa_kwargs: dict = {},
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    ''' Function to generate one of our ResNet models and respective optimizer.
        
        Optimizers will be SGD, with optional Stochastic Weight Averaging. 
        For more on SWA, see https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/

    Args:
        model_name (str): Name of model to call from aim_perception.ModelFactory
        model_kwargs (dict): Keyword arguments for models supported in aim_perception.ModelFactory
        learning_rate (float): Learning rate for optimizer. Default,  1e-1.
        weight_decay (float): Weight decay for optimizer. Default, 1e-4, 
        momentum (float): Momentum for optimizer. Default, 0.9.
        swa_kwargs (dict): Keyword arguments for torchcontrib.optim.SWA. If None, SWA wont be used. Default, None.

    Returns:    
        model (nn.Module): Model generated from aim_perception.ModelFactory
        optimizer (torch.optim.Optimizer): Torch optimizer for training
    '''
    # Create model
    model = ModelFactory.get_model(model_name, **model_kwargs)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum
    )

    if swa_kwargs:
        swa = torchcontrib.optim.SWA(optimizer, **swa_kwargs)

        swa.state = optimizer.state
        swa.defaults = optimizer.defaults
        swa.param_groups = optimizer.param_groups

        return model, swa

    return model, optimizer

def train_model(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    wandb_project: str,
    wandb_run_name: str,
    epochs: int = 45,
    fine_tune_epochs: int = 15,
    scheduler_step_size: int = 15,
    scheduler_gamma: float = 0.1,
    save_path: str = None,
    wandb_config_updates: dict = {},
) -> nn.Module:
    ''' Function for a wrapping a model training run.
        - Criterion fixed to CrossEntropy loss with class weights
        - Uses a step scheduler for learning rate
        - Optional wandb integration for reporting results
        - Will print evaluation metrics on validation set

    Args:
        model (nn.Module): Torch model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        train_loader (torch.utils.data.DataLoader): Training split data loader with AimDataset
        val_loader (torch.utils.data.DataLoader): Validation split data loader with AimDataset
        wandb_project (str): WandB project to report results to (required for sweeping)
        wandb_run_name (str): WandB run name
        epochs (int): Epochs to train for. Default, 45.
        fine_tune_epochs (int): Number of epochs to freeze back bone for. Default, 15.
        scheduler_step_size (int): Step size of step learning rate scheduler. Default, 15.
        scheduler_gamma (float): Scheduler gamma of step learning rate scheduler. Default, . 0.1.
        save_path (str): Optional path to save model to. Default, None.
        wandb_config_updates (dict): Optional updates for wandb config. Default, {}.

    Returns:
        model (nn.Module): Trained model
    '''

    # Instantiate Loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights = train_loader.dataset.get_class_weights().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Instantiate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )

    # Create Trainer
    trainer = Trainer(
        epochs=epochs, 
        validate_every=100, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        save_path=save_path,
        wandb_project=wandb_project,
        wandb_config_updates=wandb_config_updates,
    )

    eval = trainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        run_name=wandb_run_name,
        fine_tune_epochs=fine_tune_epochs
    )

    print(eval.classification_report)

    return model
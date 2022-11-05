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

# Configure data loading
IMAGE_MEAN = [0.4886, 0.4855, 0.4838]
IMAGE_STD = [0.2456, 0.2443, 0.2490]


def create_data_loaders(root_data_path: str, image_size: Tuple[int], batch_size: int) -> Tuple[torch.utils.data.DataLoader]:
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
    momentum=0.9,
    swa_kwargs: dict = {},
) -> Tuple[nn.Module, torch.optim.Optimizer]:

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
    epochs: int = 30,
    fine_tune_epochs: int = 10,
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    save_path: str = None,
    wandb_config_updates: dict = {},
) -> nn.Module:

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
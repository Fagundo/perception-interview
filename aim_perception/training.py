import wandb
import torch
from torch import nn
from datetime import datetime
from aim_perception.loading import AimDataset
from aim_perception.evaluation import Evaluation


class Trainer:

    def __init__(
        self, 
        epochs: int,
        validate_every: int, 
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        wandb_project: str = None,
    ) -> None:

        self._epochs = epochs
        self._validate_every = validate_every
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._wandb = wandb_project

        
    def config_wandb(self, fine_tune_epochs: int, train_loader: torch.utils.data.DataLoader):
        wandb.init(
            project=self._wandb,
            name=datetime.utcnow().isoformat(),
            config=dict(
                epochs=self._epochs,
                batch_size=train_loader.batch_size,
                fine_tune_epochs=fine_tune_epochs,
                learning_rate=self._optimizer.defaults['lr'],
                weight_decay=self._optimizer.defaults['weight_decay'],
            )
        )

    def run_eval(
        self, 
        epoch_num: int, 
        batch_num: int, 
        running_loss: float, 
        model: nn.Module, 
        val_loader: AimDataset
    ) -> None:

        # Instantiate val loss
        running_val_loss = 0

        with torch.no_grad():
            
            y_val_accum, y_val_hat_accum = [], []
            
            # Run over validation and calculate losses
            for j, val_batch in enumerate(val_loader):
                x_val, y_val = val_batch

                # Get val estimate
                y_val_hat = model(x_val)

                # Calculate loss
                loss = self._criterion(y_val_hat, y_val)
                running_val_loss += loss

                # Accumulate validation
                y_val_accum.append(y_val)
                y_val_hat_accum.append(y_val_hat)

        # Gather metrics
        epoch_train_loss = running_loss / self._validate_every
        epoch_val_loss = running_val_loss / (j + 1)

        evaluation = Evaluation(true_prob=y_val_accum, predicted_prob=y_val_hat_accum)
        balanced_accuracy = evaluation.balanced_accuracy
        
        print(f'[{epoch_num + 1}, {batch_num + 1:5d}] train loss: {epoch_train_loss:.3f} | val loss: {epoch_val_loss:.3f} | val bal-acc: {balanced_accuracy:.3f}')

        if self._wandb:
            wandb.log(dict(train_loss=epoch_train_loss, val_loss=epoch_val_loss, val_accuracy=balanced_accuracy))

    def __call__(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        fine_tune_epochs: int = None
    ) -> nn.Module:

        # Configure weights and biases if given
        if self._wandb:
            self.config_wandb(fine_tune_epochs=fine_tune_epochs, train_loader=train_loader)

        # Freeze backbone
        if fine_tune_epochs:
            model.freeze_backbone()
                    
        for epoch in range(self._epochs):  # loop over the dataset multiple times
                
            running_loss = 0.0

            if fine_tune_epochs and epoch==fine_tune_epochs:
                model.unfreeze_backbone()

            for i, batch in enumerate(train_loader, 0):
                # Retrieve inpute and labels
                x, y = batch

                # Zero out gradients
                self._optimizer.zero_grad()

                # Forward pass
                y_hat = model(x)

                # Calculate loss
                loss = self._criterion(y_hat, y)

                # Back prop
                loss.backward()
                self._optimizer.step()

                # Add loss
                running_loss += loss.item()
                
                # Validate and log loss
                if (i != 0) and (i % self._validate_every == 0):    
                    self.run_eval(
                        epoch_num=epoch, batch_num=i, running_loss=running_loss, model=model, val_loader=val_loader
                    )

                    running_loss = 0.0

        # Run batch eval
        print(f'-------- Finished Epoch {epoch + 1} --------')
        self.run_eval(
            epoch_num=epoch, batch_num=i, running_loss=running_loss, model=model, val_loader=val_loader
        )                    

        # Step the scheduler
        if self._scheduler:
            self._scheduler.step()


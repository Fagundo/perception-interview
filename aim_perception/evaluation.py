import torch
import pandas as pd
from torch import nn
from typing import Tuple, List
from sklearn.metrics import classification_report, balanced_accuracy_score


class Evaluation:
    ''' Class for handling multiclassification evaluation. 
        Supports a balanced_accuracy score and classification report to break out performance by class.

    Attributes:
        true (List[str]): List of true class labels
        pred (List[str]): List of predicted class labels
    '''
    def __init__(self, true_prob: torch.Tensor, predicted_prob: torch.Tensor, data_loader: torch.utils.data.DataLoader) -> None:
        '''
        Args:
            true_prob (torch.Tensor): True class probabilities (one hot encoded labels)
            predicted_prob (torch.Tensor): Predicted class probabilities
            data_loader (torch.utils.data.DataLoader): Data loader used to generate the input probabilities
        '''

        self.true, self.pred = self.convert_to_classes(true_prob, predicted_prob, data_loader)

    def convert_to_classes(self, true_prod: torch.Tensor, predicted_prob: torch.Tensor, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor]:
        ''' Method to convert probabilities to class labels
        Args:
            true_prob (torch.Tensor): True class probabilities (one hot encoded labels)
            predicted_prob (torch.Tensor): Predicted class probabilities
            data_loader (torch.utils.data.DataLoader): Data loader used to generate the input probabilities

        Returns:
            List[str]: List of true class labels
            List[str]: List of predicted class labels
        '''

        class_true = torch.concat(true_prod).argmax(axis=1)
        class_pred = torch.concat(predicted_prob).argmax(axis=1)

        reverse_mapping = data_loader.dataset.reverse_mapping
        class_true = [reverse_mapping[int(i)] for i in class_true]
        class_pred = [reverse_mapping[int(i)] for i in class_pred]

        return class_true, class_pred 

    @property
    def classification_report(self) -> str:
        '''Returns the sklearn.metrics.classification_report from true and predicted labels'''

        return classification_report(self.true, self.pred)

    @property
    def balanced_accuracy(self) -> str:
        '''Returns the sklearn.metrics.balanced_accuracy_score report from true and predicted labels'''

        return balanced_accuracy_score(self.true, self.pred)


class InferenceEvaluation(Evaluation):
    ''' Wrapper around Evaluation class that runs inference from model and data loader.
        Additionally supports exported results to a pandas dataframe.

    Attributes:
        true (List[str]): List of true class labels
        pred (List[str]): List of predicted class labels
        _model (nn.Module): Torch model to use for infering probablities and labels
        _loader (torch.utils.data.DataLoader): Data loader to create labels with
    '''
    def __init__(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> None:
        '''
        Args:
            model (nn.Module): Torch model to use for infering probablities and labels
            loader (torch.utils.data.DataLoader): Data loader to create labels with
        '''
        y_true_prob, y_pred_prob = self._infer(model, loader)
        super().__init__(y_true_prob, y_pred_prob, loader)

        self._model = model
        self._loader = loader

    def get_label_df(self) -> pd.DataFrame:
        ''' Returns the dataset ground truth dataframe with a column for predicted labels.

        Returns:
            df (pd.DataFrame): Pandas dataframe with pred_label column with predicted labels.
        '''
        df = self._loader.dataset._gt_df
        df['pred_label'] = self.pred

        return df

    def _infer(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> Tuple[List]:
        ''' Runs inference over the data from the data loader with the given model.
        
        Args:
            model (nn.Module): Torch model to use for infering probablities and labels
            loader (torch.utils.data.DataLoader): Data loader to create labels with

        Returns:
            y_val_accum (List[torch.Tensor]): List of true class probabilities (one hot encoded labels)
            y_val_hat_accum (List[torch.Tensor]): List of predicted class probabilities
        '''
        model.eval()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y_val_accum, y_val_hat_accum = [], []

        with torch.no_grad():
            for batch in loader:
                x_val, y_val = batch

                # To device
                x_val, y_val = x_val.to(device), y_val.to(device)

                # Get val estimate
                y_val_hat = model(x_val)

                # To Cpu
                y_val = y_val.cpu()
                y_val_hat = y_val_hat.cpu()

                # Accumulate validation
                y_val_accum.append(y_val)
                y_val_hat_accum.append(y_val_hat)

        return y_val_accum, y_val_hat_accum
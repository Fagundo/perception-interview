import torch
from torch import nn
from typing import Tuple
from sklearn.metrics import classification_report, balanced_accuracy_score


class Evaluation:

    def __init__(self, true_prob: torch.Tensor, predicted_prob: torch.Tensor, data_loader: torch.utils.data.DataLoader) -> None:

        self.true, self.pred = self.convert_to_classes(true_prob, predicted_prob, data_loader)

    def convert_to_classes(self, true_prod: torch.Tensor, predicted_prob: torch.Tensor, data_loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor]:
        class_true = torch.concat(true_prod).argmax(axis=1)
        class_pred = torch.concat(predicted_prob).argmax(axis=1)

        reverse_mapping = data_loader.dataset.reverse_mapping
        class_true = [reverse_mapping[int(i)] for i in class_true]
        class_pred = [reverse_mapping[int(i)] for i in class_pred]

        return class_true, class_pred 

    @property
    def classification_report(self) -> str:

        return classification_report(self.true, self.pred)

    @property
    def balanced_accuracy(self) -> str:

        return balanced_accuracy_score(self.true, self.pred)


class InferenceEvaluation(Evaluation):

    def __init__(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> None:
        y_true_prob, y_pred_prob = self._infer(model, loader)
        super().__init__(y_true_prob, y_pred_prob, loader)

        self._model = model
        self._loader = loader

    def get_label_df(self):
        df = self._loader.dataset._gt_df
        df['pred_label'] = self.pred

        return df

    def _infer(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> None:

        model.eval()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y_val_accum, y_val_hat_accum = [], []

        for val_batch in loader:
            x_val, y_val = val_batch

            # To device
            x_val, y_val = x_val.to(device), y_val.to(device)

            # Get val estimate
            y_val_hat = model(x_val)

            # Accumulate validation
            y_val_accum.append(y_val.cpu())
            y_val_hat_accum.append(y_val_hat.cpu())

        return y_val_accum, y_val_hat_accum
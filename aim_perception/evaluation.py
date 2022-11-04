import torch
from typing import Tuple
from sklearn.metrics import classification_report, balanced_accuracy_score


class Evaluation:

    def __init__(self, true_prob: torch.Tensor, predicted_prob: torch.Tensor) -> None:

        self.true, self.pred = self.convert_to_classes(true_prob, predicted_prob)

    def convert_to_classes(self, true_prod: torch.Tensor, predicted_prob: torch.Tensor) -> Tuple[torch.Tensor]:
        class_true = torch.concat(true_prod).argmax(axis=1)
        class_pred = torch.concat(predicted_prob).argmax(axis=1)  

        return class_true, class_pred      

    @property
    def classification_report(self) -> str:

        return classification_report(self.true, self.pred)

    @property
    def balanced_accuracy(self) -> str:

        return balanced_accuracy_score(self.true, self.pred)
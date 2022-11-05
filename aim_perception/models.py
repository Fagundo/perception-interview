import torch
import torch.nn as nn
from typing import List


class MultiClassMlpHead(nn.Module):

    def __init__(
        self, 
        input_size: int, 
        inner_dim: int,  
        num_targets: int, 
        bias: bool = True,
        dropout: float = 0.1, 
        norm: nn.Module = None,
    ) -> None:
        super().__init__()

        self._activation = nn.Softmax(dim=1)
        self._relu = nn.ReLU(inplace=True)
        self._dropout = nn.Dropout(dropout) if dropout else None
        
        self._fc1 = nn.Linear(input_size, inner_dim, bias=bias)
        self._fc2 = nn.Linear(inner_dim, num_targets, bias=bias)

        self._norm_1 = norm(inner_dim) if norm else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # First fully connected
        x = self._fc1(x)
        x = self._norm_1(x)
        x = self._relu(x)

        if self._dropout:
            x = self._dropout(x)

        # Second fully connected fully connected
        x = self._fc2(x)

        return self._activation(x)


class ModelWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self._backbone = backbone
        self._head = head

    def freeze_backbone(self):
        for param in self._backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self._backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self._backbone(x)
        x = self._head(x)

        return x
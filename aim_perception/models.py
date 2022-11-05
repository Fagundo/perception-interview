import torch
import torch.nn as nn
from typing import List


class MultiClassMlpHead(nn.Module):
    ''' Multi-Class classification multi-layer perceptron head.

    Attributes:
        _activation (nn.Softmax): Softmax activation for multiclass
        _relu (nn.ReLU): Relu activation for intermediate fully connected layer
        _dropout (nn.Dropout): Dropout for regularization at the final fully connected layer
        _fc1 (nn.Linear): First fully connected layer
        _fc2 (nn.Linear): Final fully connected layer that outputs class probabilities
        _norm_1 (nn.Module): Normalization layer for first fully connected output
    '''
    def __init__(
        self, 
        input_size: int, 
        inner_dim: int,  
        num_targets: int, 
        bias: bool = True,
        dropout: float = 0.05, 
        norm: nn.Module = None,
    ) -> None:
        '''
        Args:
            input_size (int): Dimmensionality of input tensor
            inner_dim (int): Size of inner fully connected layer
            num_targets (int): Number of targets to estimate for
            bias (bool): Whether to include bias in linear layers. Default, True.
            dropout (float): Dropout probability for last fully connected. Default, 0.05.
            norm (nn.Module): Normalization layer for first fully connected output. Default, None.
        '''
        super().__init__()

        self._activation = nn.Softmax(dim=1)
        self._relu = nn.ReLU(inplace=True)
        self._dropout = nn.Dropout(dropout) if dropout else None
        
        self._fc1 = nn.Linear(input_size, inner_dim, bias=bias)
        self._fc2 = nn.Linear(inner_dim, num_targets, bias=bias)

        self._norm_1 = norm(inner_dim) if norm else lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Torch forward pass. Outputs class probability prediction'''
        
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
    ''' Wrapper around backbone and regression head. Enables us to freeze pretrained backbones.

    Attributes:
        _backbone (nn.Module): Model backbone to encode image
        _head (nn.Module): Model prediction head
    '''
    def __init__(self, backbone: nn.Module, head: nn.Module):
        '''
        Attributes:
            backbone (nn.Module): Model backbone to encode image
            head (nn.Module): Model prediction head
        '''
        super().__init__()
        self._backbone = backbone
        self._head = head

    def freeze_backbone(self):
        ''' Method to freeze backbone '''
        for param in self._backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        ''' Method to unfreeze backbone '''
        for param in self._backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' Torch forward pass. Takes an image and outputs class probability prediction'''

        x = self._backbone(x)
        x = self._head(x)

        return x
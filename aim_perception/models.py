import torch
import torch.nn as nn


class LinearHead(nn.Module):

    def __init__(self, input_size: int, num_targets: int, dropout: float = 0.01, activation: nn.Module = nn.Softmax) -> None:
        super().__init__()
        self._activation = activation
        self._dropout = nn.Dropout(dropout) if dropout else None
        self._fc1 = nn.Linear(input_size, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self._dropout:
            x = self._dropout(x)

        return self._fc1(x)


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


### Some home made CNNs
import torch
from torch import nn


class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        
        self._depthwise_conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,
            bias=bias
        )

        self._pointwise_conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=bias
        )        

        # Enforcing kaiming (He) uniform initialization
        torch.nn.init.kaiming_uniform_(self._depthwise_conv.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self._pointwise_conv.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._depthwise_conv(x)
        out = self._pointwise_conv(out)

        return out
        

class ResNetBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        padding: int = 1, 
    ) -> None:
        super().__init__()
        
        # Activation
        self._relu = nn.ReLU(inplace=True)

        # Norm layers
        self._norm_1 = nn.BatchNorm2d(out_channels)
        self._norm_2 = nn.BatchNorm2d(out_channels)

        # Conv layers
        self._conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False
        )

        self._conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False
        )

        # Residual projections
        if (stride != 1) or (in_channels != out_channels):
             # Make convolution and kaiming (He) init
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            torch.nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            
            self._res_projection = nn.Sequential(conv, nn.BatchNorm2d(out_channels))

        else:
            self._res_projection = lambda x: x  # Just pass me along

        # Enforcing kaiming (He) uniform initialization
        torch.nn.init.kaiming_uniform_(self._depthwise_conv.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self._pointwise_conv.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # First conv + norm + activation
        out = self._conv_1(x)
        out = self._norm_1(out)
        out = self._relu(out)

        # Second conv + norm + residual + activation
        out = self._conv_2(out)
        out = self._norm_2(out)

        # Projection of residual
        res = self._res_projection(x)

        return self._relu(out + res)

class ResNetBlockSeparable(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        padding: int = 1, 
    ) -> None:
        super().__init__()
        
        # Activation
        self._relu = nn.ReLU(inplace=True)

        # Norm layers
        self._norm_1 = nn.BatchNorm2d(out_channels)
        self._norm_2 = nn.BatchNorm2d(out_channels)

        # Conv layers
        self._conv_1 = SeparableConvBlock(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False
        )

        self._conv_2 = SeparableConvBlock(
            out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False
        )

        # Residual projections
        if (stride != 1) or (in_channels != out_channels):
            self._res_projection = nn.Sequential(
                SeparableConvBlock(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self._res_projection = lambda x: x  # Just pass me along

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # First conv + norm + activation
        out = self._conv_1(x)
        out = self._norm_1(out)
        out = self._relu(out)

        # Second conv + norm + residual + activation
        out = self._conv_2(out)
        out = self._norm_2(out)

        # Projection of residual
        res = self._res_projection(x)

        return self._relu(out + res)


class ResNetBrick(nn.Module):
    ''' Calling this a brick because its bigger than a block.
    '''
    def __init__(
        self, 
        in_channels: int, 
        inner_channels: int, 
        downsample_stride: int, 
        num_layers: int, 
        depthwise_separable: bool = False
    ) -> None:
        super().__init__()
        
        if depthwise_separable:
            res_block = ResNetBlockSeparable
        else:
            res_block = ResNetBlock

        self.layers = self._make_layers(
            in_channels, inner_channels, downsample_stride, num_layers, res_block
        )

    def _make_layers(
        self, 
        in_channels: int, 
        inner_channels: int, 
        downsample_stride: int, 
        num_layers: int, 
        res_block: nn.Module
    ) -> nn.Module:
        
        layers = []
        for layer_num in range(num_layers):

            # Initial Layer where we change in channels to inner channels
            if layer_num==0:
                layer = res_block(in_channels, inner_channels)
                
            elif layer_num==num_layers - 1:
                layer = res_block(inner_channels, inner_channels, stride=downsample_stride)
            
            else: 
                layer = res_block(inner_channels, inner_channels)

            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResNet34(nn.Module):

    def __init__(self, in_channels: int, depthwise_separable: bool = False):
        super().__init__()
        '''
        Note, this is different from true resnet, with kernel_size = 3 instead of 7.
        I chose this because the size of the images given are much smaller in our dataset
        than in imagenet. Thus, I also remove the first max pool.
        '''
        self._conv_1 = nn.Conv2d(kernel_size=3, in_channels=3, out_channels=64, stride=2, bias=False)
        self._norm_1 = nn.BatchNorm2d(64)
        self._activation_1 = nn.ReLU()

        # Instantiate convs similar to resnet 34
        self._conv_2 = ResNetBrick(
            in_channels=64, 
            inner_channels=64, 
            downsample_stride=2, 
            num_layers=3, 
            depthwise_separable=depthwise_separable
        )

        self._conv_3 = ResNetBrick(
            in_channels=64, 
            inner_channels=128, 
            downsample_stride=2, 
            num_layers=4, 
            depthwise_separable=depthwise_separable
        )       

        self._conv_4 = ResNetBrick(
            in_channels=128, 
            inner_channels=256, 
            downsample_stride=2, 
            num_layers=6, 
            depthwise_separable=depthwise_separable
        )        

        self._conv_5 = ResNetBrick(
            in_channels=256, 
            inner_channels=512, 
            downsample_stride=2, 
            num_layers=3, 
            depthwise_separable=depthwise_separable
        )        

        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # adaptive average pool to give [512, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self._conv_1(x)
        out = self._norm_1(out)
        out = self._activation_1(out)

        out = self._conv_2(out)
        out = self._conv_3(out)
        out = self._conv_4(out)
        out = self._conv_5(out)

        out = self._avg_pool(out)

        out = torch.flatten(out, 1) 

        # No fully connected, leaving that up to the heads

        return out
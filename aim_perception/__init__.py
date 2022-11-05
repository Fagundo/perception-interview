from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from aim_perception.models import ModelWrapper, MultiClassMlpHead


class ModelFactory:
    '''A factory method for easily calling models based on the resnset size'''

    @classmethod
    def _resnet_18(cls, dropout: int, inner_dim: int = 100) -> nn.Module:
        '''Returns a ResNet18 with ImageNet pretrained weights
        
        Args:
            dropout (int): Dropout rate to apply to the MLP head.
            inner_dim (int): Inner dimmension of the MLP head.

        Returns:
            nn.Module: Torch nn.Module with pretrained Resnet18 backbone and MLP head.
        '''
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        head = MultiClassMlpHead(
            input_size=1000, inner_dim=inner_dim, num_targets=10, dropout=dropout, norm=nn.BatchNorm1d
        )
        return ModelWrapper(backbone=backbone, head=head)

    @classmethod
    def _resnet_34(cls, dropout: int, inner_dim: int = 100) -> nn.Module:
        '''Returns a ResNet34 with ImageNet pretrained weights
        
        Args:
            dropout (int): Dropout rate to apply to the MLP head.
            inner_dim (int): Inner dimmension of the MLP head.

        Returns:
            nn.Module: Torch nn.Module with pretrained ResNet34 backbone and MLP head.
        '''

        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        head = MultiClassMlpHead(
            input_size=1000, inner_dim=inner_dim, num_targets=10, dropout=dropout, norm=nn.BatchNorm1d
        )

        return ModelWrapper(backbone=backbone, head=head)

    @classmethod
    def _resnet_50(cls, dropout: int, inner_dim: int = 100) -> nn.Module:
        '''Returns a ResNet50 with ImageNet pretrained weights
        
        Args:
            dropout (int): Dropout rate to apply to the MLP head.
            inner_dim (int): Inner dimmension of the MLP head.

        Returns:
            nn.Module: Torch nn.Module with pretrained ResNet50 backbone and MLP head.
        '''        
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        head = MultiClassMlpHead(
            input_size=1000, inner_dim=inner_dim, num_targets=10, dropout=dropout, norm=nn.BatchNorm1d
        )

        return ModelWrapper(backbone=backbone, head=head)        

    @classmethod
    def get_model(cls, model_name: str, **model_kwargs):

        if model_name=='resnet_18': 
            return cls._resnet_18(**model_kwargs)

        elif model_name=='resnet_34': 
            return cls._resnet_34(**model_kwargs)

        elif model_name=='resnet_50': 
            return cls._resnet_50(**model_kwargs)            

        else:
            raise Exception(f'Model type {model_name} not supported!')

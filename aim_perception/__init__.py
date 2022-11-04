from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from aim_perception.models import ResNet, ModelWrapper, MultiClassMlpHead


class ModelFactory:

    @classmethod
    def _resnet_18_image_net(cls, dropout: int) -> nn.Module:
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        head = MultiClassMlpHead(
            input_size=1000, inner_dim=100, num_targets=10, dropout=dropout, norm=nn.BatchNorm1d
        )
        return ModelWrapper(backbone=backbone, head=head)

    @classmethod
    def _resnet_34_image_net(cls, dropout: int) -> nn.Module:
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        head = MultiClassMlpHead(
            input_size=1000, inner_dim=100, num_targets=10, dropout=dropout, norm=nn.BatchNorm1d
        )

        return ModelWrapper(backbone=backbone, head=head)

    @classmethod
    def _resnet_18(cls, dropout: int, depthwise: bool) -> nn.Module:
        backbone = ResNet.resnet_18(in_channels=3, depthwise_separable=depthwise)
        head = MultiClassMlpHead(
            input_size=512, inner_dim=256, num_targets=10, dropout=dropout, norm=nn.BatchNorm1d
        )

        return ModelWrapper(backbone=backbone, head=head)

    @classmethod
    def _resnet_34(cls, dropout: int, depthwise: bool) -> nn.Module:
        backbone = ResNet.resnet_18(in_channels=3, depthwise_separable=depthwise)
        head = MultiClassMlpHead(
            input_size=512, inner_dim=256, num_targets=10, dropout=dropout, norm=nn.BatchNorm1d
        )

        return ModelWrapper(backbone=backbone, head=head)

    @classmethod
    def get_model(cls, model_name: str, **model_kwargs):

        if model_name=='resnet_18_imagenet': 
            return cls._resnet_18_image_net(**model_kwargs)

        elif model_name=='resnet_34_imagenet': 
            return cls._resnet_34_image_net(**model_kwargs)

        elif model_name=='resnet_18':             
            return cls._resnet_18(**model_kwargs)

        elif model_name=='resnet_34':                         
            return cls._resnet_34(**model_kwargs)

        else:
            raise Exception(f'Model {model_name} not supported')

from .resnet import *
from .MobileNetV3 import MobileNetV3
from .ShuffleNetV2 import *

__all__ = ['get_backbone']

support_backbone = ['resnet18', 'deformable_resnet18', 'deformable_resnet50',
                    'resnet50', 'resnet34', 'resnet101', 'resnet152',
                    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                    'MobileNetV3']


def get_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'{backbone_name} dose not suppoted, all support backbone are include {support_backbone}'
    
    return eval(backbone_name)(**kwargs)


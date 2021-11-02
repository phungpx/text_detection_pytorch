import torch
import torch.nn as nn 
from .backbones import get_backbone
from .head import get_head
from .neck import get_neck
from collections import OrderedDict


class DBModel(nn.Module):

    def __init__(self, backbone_name, config_backbone, config_neck, config_head):
        super().__init__()

        self.backbone = get_backbone(backbone_name, **config_backbone)
        self.neck = get_neck(**config_neck)
        self.head = get_head(**config_head)
    
    def forward(self, inputs):
        _, _, H, W = inputs.size()
        backbone_out = self.backbone(inputs)
        features = self.neck(backbone_out)
        head_out = self.head(features)
        return nn.functional.interpolate(head_out, size=(H, W), mode='bilinear', align_corners=True)

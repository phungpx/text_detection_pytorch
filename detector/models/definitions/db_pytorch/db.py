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


if __name__ == '__main__':
    backbone_name = 'resnet50'
    config_backbone = {
        'pretrained': True,
        'in_channels': 3
    }
    config_head = {
        'in_channels': 256
    }
    config_extractor = {
        'in_channels': [256, 512, 1024, 2048],
        'inner_channels': 256
    }
    model = DBModel(backbone_name, config_backbone, config_extractor, config_head)
    weight  = torch.load('models/weights/db/ic15_resnet50', map_location='cpu')

    bb_weight = OrderedDict()
    neck_weight = OrderedDict()
    head_weight = OrderedDict()
    
    for module_name in weight:
        if 'backbone' in module_name and '_offset' not in module_name:
           bb_weight[module_name[22:]] = weight[module_name]
        
        elif 'binarize' in module_name or 'thresh' in module_name:
            head_weight[module_name[21:]] = weight[module_name]
        elif 'decoder.in' in module_name or 'decoder.out' in module_name:
            neck_weight[module_name[21:]] = weight[module_name]
        
    bb_weight.pop("fc.weight")
    bb_weight.pop("fc.bias")
    bb_weight.pop("smooth.weight")
    bb_weight.pop("smooth.bias")
    model.backbone.load_state_dict(bb_weight)
    model.head.load_state_dict(head_weight)
    model.neck.load_state_dict(neck_weight)

    x = torch.randn(1, 3, 448, 448)
    y = model(x)
    print(y.shape)
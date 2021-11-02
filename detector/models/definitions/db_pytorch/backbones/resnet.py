from typing import List, Optional
import torch.nn as nn 
import math
import torch.utils.model_zoo as model_zoo 
from torchvision.ops import DeformConv2d


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
           'deformable_resnet18', 'deformable_resnet50', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def constant_init(module, constant, bias=0):
    nn.init.constant_(module.weight, constant)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None, 
                 dcn=None):

        super().__init__()
        self.with_dcn = dcn is not None 
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        if not self.with_dcn:
            self.conv2 = nn.Conv2d(in_channels=out_channels, 
                                   out_channels=out_channels,
                                   kernel_size=3, padding=1, 
                                   bias=False)
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(in_channels=out_channels, 
                                          out_channels=deformable_groups * offset_channels,
                                          kernel_size=3,
                                          padding=1)
            self.conv2 = DeformConv2d(in_channels=out_channels,
                                      out_channels=out_channels, 
                                      kernel_size=3,
                                      padding=1,
                                      bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample
        self.stride = stride 

    def forward(self, inputs):
        residual = inputs
        out = self.relu(self.bn1(self.conv1(inputs)))
        
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)




class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels, 
                 out_channels, 
                 stride=1, 
                 downsample=None, 
                 dcn=None):

        super().__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, 
                               kernel_size=1, 
                               bias=False
                            )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(
                                    in_channels=out_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=3, 
                                    stride=stride, 
                                    padding=1, 
                                    bias=False
                                )
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            from torchvision.ops import DeformConv2d
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(
                                        in_channels=out_channels, 
                                        out_channels=deformable_groups * offset_channels, 
                                        stride=stride, 
                                        kernel_size=3, 
                                        padding=1
                                    )
            self.conv2 = DeformConv2d(in_channels=out_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=3, 
                                      padding=1, 
                                      stride=stride, 
                                      bias=False
                                    )

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels * 4, 
                               kernel_size=1, 
                               bias=False
                            )
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers, 
                 in_channels=3, 
                 dcn=None
            ):
        self.dcn = dcn
        self.inplanes = 64
        super().__init__()
        self.out_channels = []
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=64, 
                               kernel_size=7, 
                               stride=2, 
                               padding=3,
                               bias=False
                            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, BottleNeck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self,
                    block, 
                    planes,
                     blocks, 
                     stride=1, 
                     dcn=None
                    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes,
                          out_channels=planes * block.expansion,
                          kernel_size=1, 
                          stride=stride, 
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))
        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5



def resnet18(pretrained=True, **kwargs):
    
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)

    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must 3 when pre-trained is True'
        print('Loading weight from ImageNet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

    return model
    

def deformable_resnet18(pretrained=True, **kwargs):

    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], dcn=dict(deformable_groups=1),**kwargs)

    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must 3 when pre-trained is True'
        print('Loading weight from ImageNet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)

    return model

def resnet34(pretrained=True, **kwargs):
    
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)

    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must 3 when pre-trained is True'
        print('Loading weight from ImageNet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)

    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BottleNeck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must be 3 whem pretrained is True'
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def deformable_resnet50(pretrained=True, **kwargs):

    model = ResNet(block=BottleNeck, layers=[3, 4, 6, 3], dcn=dict(deformable_groups=1),**kwargs)

    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must 3 when pre-trained is True'
        print('Loading weight from ImageNet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)

    return model

def resnet101(pretrained=True, **kwargs):

    model = ResNet(block=BottleNeck, layers=[3, 4, 23, 3], **kwargs)

    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must 3 when pre-trained is True'
        print('Loading weight from ImageNet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)

    return model

def resnet152(pretrained=True, **kwargs):

    model = ResNet(block=BottleNeck, layers=[3, 8, 36, 3], **kwargs)

    if pretrained:
        assert kwargs['in_channels'] == 3, 'in_channels must 3 when pre-trained is True'
        print('Loading weight from ImageNet')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)

    return model


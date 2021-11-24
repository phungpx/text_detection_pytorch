from torch import nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, backbone_name, pretrained=False):
        super(ResNet, self).__init__()
        backbone = getattr(models, backbone_name)(pretrained)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)  # c2

        x = self.layer2(x)
        features.append(x)  # c3

        x = self.layer3(x)
        features.append(x)  # c4

        x = self.layer4(x)
        features.append(x)  # c5

        return features

from torch import nn
from .resnet import ResNet
from .seg_head import FPEM_FFM


resnet_out_channels = {
    'resnet18': [64, 128, 256, 512],
    'resnet34': [64, 128, 256, 512],
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048],
}


class PANNet(nn.Module):
    def __init__(
        self,
        backbone_name='resnet18',
        backbone_pretrained=False,
        num_FPEMs=2,
    ):
        super(PANNet, self).__init__()
        self.resnet = ResNet(
            backbone_name=backbone_name,
            pretrained=backbone_pretrained
        )

        self.seg_head = FPEM_FFM(
            backbone_out_channels=resnet_out_channels[backbone_name],
            num_FPEMs=num_FPEMs
        )

    def forward(self, x):
        H, W = x.shape[2:]

        features = self.resnet(x)  # c2, c3, c4, c5
        x = self.seg_head(features)  # N x 6 x H / 4, W / 4
        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)  # N x 6 x H x W

        return x

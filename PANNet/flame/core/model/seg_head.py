import torch
from torch import nn
from typing import List, Tuple


class SeparableConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1
    ) -> None:
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(self.bn(x))
        return x


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        return x


class FPEM(nn.Module):
    def __init__(self, in_channels: int = 128):
        super(FPEM, self).__init__()
        # upscale enhencement
        self.conv3x3_up1 = SeparableConv2D(in_channels, in_channels, kernel_size=3, stride=1)
        self.conv3x3_up2 = SeparableConv2D(in_channels, in_channels, kernel_size=3, stride=1)
        self.conv3x3_up3 = SeparableConv2D(in_channels, in_channels, kernel_size=3, stride=1)

        # downscale enhencement
        self.conv3x3_down1 = SeparableConv2D(in_channels, in_channels, kernel_size=3, stride=2)
        self.conv3x3_down2 = SeparableConv2D(in_channels, in_channels, kernel_size=3, stride=2)
        self.conv3x3_down3 = SeparableConv2D(in_channels, in_channels, kernel_size=3, stride=2)

    def upsample_add(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x1, size=x2.shape[2:], mode='bilinear') + x2
        return x

    def forward(self, features: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        c2, c3, c4, c5 = features

        # up-scale enhencement
        c4 = self.conv3x3_up1(self.upsample_add(c5, c4))
        c3 = self.conv3x3_up2(self.upsample_add(c4, c3))
        c2 = self.conv3x3_up3(self.upsample_add(c3, c2))

        # down-scale enhencement
        c3 = self.conv3x3_down1(self.upsample_add(c3, c2))
        c4 = self.conv3x3_down2(self.upsample_add(c4, c3))
        c5 = self.conv3x3_down3(self.upsample_add(c5, c4))

        return c2, c3, c4, c5


class FPEM_FFM(nn.Module):
    def __init__(self, backbone_out_channels: List[int], num_FPEMs: int = 2):
        super(FPEM_FFM, self).__init__()
        out_channels = 128

        # reducing channel layers
        self.conv1x1_c2 = ConvBNReLU(
            in_channels=backbone_out_channels[0],
            out_channels=out_channels,
            kernel_size=1,
        )

        self.conv1x1_c3 = ConvBNReLU(
            in_channels=backbone_out_channels[1],
            out_channels=out_channels,
            kernel_size=1,
        )

        self.conv1x1_c4 = ConvBNReLU(
            in_channels=backbone_out_channels[2],
            out_channels=out_channels,
            kernel_size=1,
        )

        self.conv1x1_c5 = ConvBNReLU(
            in_channels=backbone_out_channels[3],
            out_channels=out_channels,
            kernel_size=1,
        )

        # FPEM layers
        self.FPEMs = nn.ModuleList()
        for i in range(num_FPEMs):
            self.FPEMs.append(FPEM(in_channels=out_channels))

        # final layer
        self.conv_final = nn.Conv2d(
            in_channels=out_channels * 4,
            out_channels=6,
            kernel_size=1,
        )

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        '''
        Args:
            c2: Tensor, N x C2 x H / 4 x W / 4
            c3: Tensor, N x C2 x H / 8 x W / 8
            c4: Tensor, N x C2 x H / 16 x W / 16
            c5: Tensor, N x C2 x H / 32 x W / 32
        Output:
            Tensor, N x 6 x H / 4 x W / 4
        '''
        c2, c3, c4, c5 = x

        # reduce backbone channels
        c2 = self.conv1x1_c2(c2)  # N x 128 x H / 4 x W / 4
        c3 = self.conv1x1_c3(c3)  # N x 128 x H / 8 x W / 8
        c4 = self.conv1x1_c4(c4)  # N x 128 x H / 16 x W / 16
        c5 = self.conv1x1_c5(c5)  # N x 128 x H / 32 x W / 32

        # FPEMs
        for i, fpem in enumerate(self.FPEMs):
            c2, c3, c4, c5 = fpem(features=(c2, c3, c4, c5))
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm += c2
                c3_ffm += c3
                c4_ffm += c4
                c5_ffm += c5

        # FFM
        c5_ffm = nn.functional.interpolate(c5_ffm, size=c2_ffm.shape[2:], mode='bilinear')
        c4_ffm = nn.functional.interpolate(c4_ffm, size=c2_ffm.shape[2:], mode='bilinear')
        c3_ffm = nn.functional.interpolate(c3_ffm, size=c2_ffm.shape[2:], mode='bilinear')
        Fy = torch.cat([c2_ffm, c3_ffm, c4_ffm, c5_ffm], dim=1)   # N x 512 x H / 4 x W / 4

        # Final
        x = self.conv_final(Fy)  # N x 6 x H / 4 x W / 4

        return x

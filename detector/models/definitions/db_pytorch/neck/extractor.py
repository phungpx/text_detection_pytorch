import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU 
import torch.nn.functional as F


# Define basic convolution using batchnorm and relu after

class Conv2dBatchNorm2dReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 inplace=True
            ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias,
                              padding_mode=padding_mode
                            )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=inplace)
    
    def forward(self, inputs):
        return self.relu(self.bn(self.conv(inputs)))


# class Extractor(nn.Module):
#     def __init__(self,
#                  in_channels=[64, 128, 256, 512],
#                  inner_channels=256,
#                  **kwargs
#         ):
#         super().__init__()
    
#         self.out_channels = inner_channels
#         inner_channels = inner_channels // 4

        
#         # reduce layer 
#         self.reduce_conv_2 = Conv2dBatchNorm2dReLU(in_channels=in_channels[0],
#                                                    out_channels=inner_channels,
#                                                    kernel_size=1,
#                                                 )
#         self.reduce_conv_3 = Conv2dBatchNorm2dReLU(in_channels=in_channels[1],
#                                                    out_channels=inner_channels,
#                                                    kernel_size=1,
#                                                 )
#         self.reduce_conv_4 = Conv2dBatchNorm2dReLU(in_channels=in_channels[2],
#                                                    out_channels=inner_channels,
#                                                    kernel_size=1,
#                                                 )
#         self.reduce_conv_5 = Conv2dBatchNorm2dReLU(in_channels=in_channels[3],
#                                                    out_channels=inner_channels,
#                                                    kernel_size=1,
#                                                 )
        
#         # smooth layer 
#         self.smooth_4 = Conv2dBatchNorm2dReLU(in_channels=inner_channels,
#                                               out_channels=inner_channels,
#                                               kernel_size=3,
#                                               padding=1,)
#         self.smooth_3 = Conv2dBatchNorm2dReLU(in_channels=inner_channels,
#                                               out_channels=inner_channels,
#                                               kernel_size=3,
#                                               padding=1,)
#         self.smooth_2 = Conv2dBatchNorm2dReLU(in_channels=inner_channels,
#                                               out_channels=inner_channels,
#                                               kernel_size=3,
#                                               padding=1,)
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=self.out_channels, 
#                       out_channels=self.out_channels, 
#                       kernel_size=3,
#                       padding=1,
#                       stride=1),
#             nn.BatchNorm2d(num_features=self.out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, inputs):
#         in_stage_2, in_stage_3, in_stage_4, in_stage_5 = inputs

#         out_stage_5 = self.reduce_conv_5(in_stage_5)
#         out_stage_4 = self.smooth_4(self._upsample_add(out_stage_5, self.reduce_conv_4(in_stage_4)))
#         out_stage_3 = self.smooth_3(self._upsample_add(out_stage_4, self.reduce_conv_3(in_stage_3)))
#         out_stage_2 = self.smooth_2(self._upsample_add(out_stage_3, self.reduce_conv_2(in_stage_2)))

#         return self.conv(self._upsample_cat(out_stage_2, out_stage_3, out_stage_4, out_stage_5))


#     def _upsample_add(self, x, y):
#         return F.interpolate(x, size=y.size()[2:]) + y
    
#     def _upsample_cat(self, p2, p3, p4, p5):
#         h, w = p2.size()[2:]
#         p3 = F.interpolate(p3, size=(h, w))
#         p4 = F.interpolate(p4, size=(h, w))
#         p5 = F.interpolate(p5, size=(h, w))
#         return torch.cat([p2, p3, p4, p5], dim=1)

class Extractor(nn.Module):
    def __init__(self, 
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256,
                 bias=False,
                 **kwargs):
        super().__init__()

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)
    
    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        fuse = torch.cat((p5, p4, p3, p2), 1)
        return fuse

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    


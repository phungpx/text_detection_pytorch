import torch 
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, 
                 in_channels,
                 k=50
            ):
        super().__init__()
    
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 4,
                kernel_size=3,
                padding=1, 
                bias=False
            ),
            nn.BatchNorm2d(num_features=in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=in_channels // 4,
                out_channels=in_channels // 4,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(num_features=in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=in_channels // 4, 
                out_channels=1,
                kernel_size=2,
                stride=2
                ),
            nn.Sigmoid()
        )

        self.thresh = self._init_thresh(inner_channels=in_channels)

    def forward(self, inputs):
        
        shrink_map = self.binarize(inputs)
        threshold_map = self.thresh(inputs)
#        if self.training:
#            binary_map = self.step_function(shrink_map, threshold_map)
#            return torch.cat([shrink_map, threshold_map, binary_map], 1)
        # predict
        return torch.cat([shrink_map, threshold_map], 1)
    

    def _init_thresh(self,
                     inner_channels, 
                     serial=False,
                     smooth=False,
                     bias=False):

        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inner_channels // 4,
                kernel_size=3,
                padding=1,
                bias=bias
            ),
            nn.BatchNorm2d(num_features=inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(num_features=inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid()
        )
        
        return self.thresh

    def _init_upsample(self,
                       in_channels, 
                       out_channels, 
                       smooth=False, 
                       bias=False):
        if smooth:
            inter_output_channels = out_channels
            if out_channels == 1:
                inter_output_channels = in_channels
            module_list = [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=inter_output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=1,
                    bias=bias
                )
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=1,
                        bias=True
                    )
                )
            return nn.Sequential(module_list)
        return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=2, stride=2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

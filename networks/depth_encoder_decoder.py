from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from models.layers import *

class DepthEncoderDecoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthEncoderDecoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))


        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225    #[12. 6. 192. 640]
        x = self.encoder.conv1(x)           #[12. 64. 96. 320]
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))  #[12. 64. 96. 320]
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))     #[12. 64. 48. 160]
        self.features.append(self.encoder.layer2(self.features[-1]))    #[12. 128. 24. 80]
        self.features.append(self.encoder.layer3(self.features[-1]))    #[12. 256. 12. 40]
        self.features.append(self.encoder.layer4(self.features[-1]))    #[12. 512. 6. 20]

        self.outputs = {}

        # decoder
        x = self.features[-1]

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [self.features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
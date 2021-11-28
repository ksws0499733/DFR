import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.my_atention import *

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, attention = True):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            input_dim = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
            input_dim = 256
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
            input_dim = 256
        elif backbone in ['CSPnet','CSPnet-l','CSPnet-m','CSPnet-x']:
            cho = ['CSPnet','CSPnet-m','CSPnet-l','CSPnet-x']
            iid = cho.index(backbone)
            low_level_inplanes = 32*(iid+2)
            input_dim = 256
        else:
            raise NotImplementedError
        
        if attention:
            self.caL = ChannelAttention(in_planes = input_dim+48)
            self.saL = SpatialAttention(kernel_size=3)
        else:
            self.caL = NoAttention()
            self.saL = NoAttention()            

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(input_dim+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        AC = self.caL(x)
        x = AC*x
        AS = self.saL(x)
        x = AS*x
        
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm , attention = True):
    return Decoder(num_classes, backbone, BatchNorm, attention)
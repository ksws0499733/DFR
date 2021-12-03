import torch
import torch.nn as nn
import torch.nn.functional as F


from modeling.backbone import mobilenet

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone
from modeling.SRM import build_BSRM
from modeling.cbam import *


class SRNN(nn.Module):
    def __init__(self, backbone='resnet', neck='A-S', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, attention = True):
        super(SRNN, self).__init__()

        if backbone == 'resnet':        
            input_dim = 256
            low_output_dim = 256
            output_dim = 2048
        elif backbone == 'drn':
            input_dim = 256
            low_output_dim = 256
            output_dim = 512
        elif backbone == 'xception':
            input_dim = 256
            low_output_dim = 128
            output_dim = 2048
        elif backbone == 'mobilenet':
            input_dim = 256
            low_output_dim = 24
            output_dim = 320
        elif backbone in ['CSPnet','CSPnet-m','CSPnet-l','CSPnet-x']:
            cho = ['CSPnet','CSPnet-m','CSPnet-l','CSPnet-x']
            iid = cho.index(backbone)
            input_dim = 256
            low_output_dim = 32*(iid+2)
            output_dim = 256*(iid+2)
        else:
            raise NotImplementedError

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride = output_stride, BatchNorm = BatchNorm, pretrain= False)
        # self.backbone = mobilenet.MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
        self.neck1 = neck.split('-')[0]
        self.neck2 = neck.split('-')[1]
        self.at = attention
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, output_dim)
        self.decoder_cls = build_decoder(num_classes, backbone, BatchNorm, attention)
        
        self.conv1 = nn.Conv2d(output_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.orderConv = build_BSRM(BatchNorm=nn.BatchNorm2d,input_dim=input_dim,kernel_size=5)        
        self.orderConvLow = build_BSRM(BatchNorm=nn.BatchNorm2d,input_dim=low_output_dim,kernel_size=5)                
        self.freeze_bn = freeze_bn

        if attention:
            self.caH = ChannelAttention(in_planes = input_dim)
            self.saH = SpatialAttention(kernel_size=3)
        else:
            self.caH = NoAttention()
            self.saH = NoAttention()      

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        # x = self.aspp(x)
        for s in self.neck1:
            if s == 'A':
                x = self.aspp(x)
            elif s == 'S':
                x = self.orderConv(x)
            elif s == 'N':
                x = self.conv1(x)
            else:
                raise NotImplementedError

        ac1 = self.caH(x)
        x = ac1*x
        as1 = self.saH(x)
        x = as1*x
        # low_level_feat = self.orderConvLow(low_level_feat)
        for s in self.neck2:
            if s == 'S':
                low_level_feat = self.orderConvLow(low_level_feat)
            elif s == 'N':
                low_level_feat = low_level_feat
            else:
                raise NotImplementedError
        
        x = self.decoder_cls(x, low_level_feat)

        out_cls = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return  out_cls



    def subweight(self, x, wt):
        # x => N,C,H,W
        # wt=> N,1,H,W
        C = x.shape[1]  
        wt = wt.repeat(1, C, 1, 1)  #增加softmax可能更好
        return wt*x

    def freeze_bn_n(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder_cls]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p



if __name__ == "__main__":
    model = SRNN(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output['class'].shape)
    print(output['instans'].shape)


import torch
import torch.nn as nn
import torch.nn.functional as F


from modeling.backbone import mobilenet

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import Decoder
from modeling.backbone import build_backbone
from modeling.SRM import build_BSRM
from modeling.cbam import *

backbone_dic = {
    "resnet":[256, 256, 2048],
    "drn":[256, 256, 512],
    "xception":[256, 128, 2048],
    "mobilenet":[256, 24, 320],
    "CSPnet":[256, 64, 512],
    "CSPnet-m":[256, 96, 768],
    "CSPnet-l":[256, 128, 1024],
    "CSPnet-x":[256, 160, 1280],
}


class RTrackNet(nn.Module):
    def __init__(self, backbone='resnet', 
                    neck='A-S', 
                    output_stride=16, 
                    num_classes=4,
                    inChannal=4,
                    sync_bn=True, 
                    freeze_bn=False, 
                    attention = True):
        super(RTrackNet, self).__init__()


        if backbone in backbone_dic.keys():
            decoder_input_dim, low_output_dim, output_dim = backbone_dic[backbone]
        else:
            raise NotImplementedError

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone,inChannal=inChannal, 
                                        output_stride = output_stride, 
                                        BatchNorm = BatchNorm, 
                                        pretrain= False)

        self.neck =self.buildNeck(neck.split('-')[0],output_stride, BatchNorm, output_dim) 

        self.neck_low =self.buildNeck(neck.split('-')[1],output_stride, BatchNorm, low_output_dim) 

        self.decoder = Decoder(num_classes, BatchNorm,
                                inChannal=decoder_input_dim,
                                low_level_inChannal= low_output_dim, 
                                attention = attention)
        
        self.freeze_bn = freeze_bn

        self.at = CBAM(in_planes = decoder_input_dim) if attention else nn.Identity()


    def forward(self, input):
        x, low_level_feat = self.backbone(input)

        x = self.at(self.neck(x))

        low_level_feat = self.neck_low(low_level_feat)
        
        out = self.decoder(x, low_level_feat)

        out = F.interpolate(out, size=input.size()[2:], mode='bilinear', align_corners=True)
        return  out

    def buildNeck(self, type, output_stride, BatchNorm, input_dim, output_dim=None, kernel_size = 3):
        if output_dim is None:
            output_dim = input_dim
        if type == 'A':
            return build_aspp( output_stride, BatchNorm, input_dim)
        elif type == 'S':
            return build_BSRM(BatchNorm=BatchNorm,input_dim=input_dim,kernel_size=kernel_size)
        elif type == 'C':
            return nn.Conv2d(input_dim, output_dim, 
                                kernel_size=kernel_size, 
                                stride=output_stride, 
                                padding=1, bias=False)
        else:
            return nn.Identity()


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
        modules = [self.neck, self.neck_low, self.decoder]
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
    model = RTrackNet(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output['class'].shape)
    print(output['instans'].shape)


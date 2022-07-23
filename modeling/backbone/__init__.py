from modeling.backbone import resnet, xception, drn, mobilenet,CSPnet
# sys.path.append(r'/home/usr1106/DeepNets/Deeplab-upload')
# print(sys.path)
def build_backbone(backbone, output_stride, BatchNorm, inChannal=3, pretrain = True):
    print('\n**** inchannal', inChannal)
    
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm,inChannal, pretrain )
    elif backbone == 'CSPnet':

        return CSPnet.CSPnet(cfg=r'./modeling/backbone/yolov5s.yaml',ch=inChannal)
    elif backbone == 'CSPnet-l':
        return CSPnet.CSPnet(cfg=r'./modeling/backbone/yolov5l.yaml',ch=inChannal)
    elif backbone == 'CSPnet-m':
        return CSPnet.CSPnet(cfg=r'./modeling/backbone/yolov5m.yaml',ch=inChannal)
    elif backbone == 'CSPnet-x':
        return CSPnet.CSPnet(cfg=r'./modeling/backbone/yolov5x.yaml',ch=inChannal)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm,inChannal, pretrain )
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm,inChannal, pretrain )
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride=output_stride, BatchNorm=BatchNorm,inChannal=inChannal, pretrained = pretrain)
    else:
        raise NotImplementedError

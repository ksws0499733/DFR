import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import sys
# sys.path.append(r'D:\论文相关\pytorch-deeplab-xception-master')

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
 


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):#分离卷积（分组卷积+1×1卷积）1

    
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x



class VGG16(nn.Module):
    
    
    def __init__(self,pretrained=True):
        super(VGG16, self).__init__()
        
        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3) # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 64 * 112 * 112
        
        self.conv2_1 = nn.Conv2d(64, 128, 3) # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56
        
        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28
        
        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14
        
        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        # self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7
        
        # # view
        
        # self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 1000)
        # # softmax 1 * 1 * 1000
         # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()
        
    def forward(self, x):
        
        # x.size(0)即为batch_size
        # in_size = x.size(0)
        
        out = self.conv1_1(x) # 222
        out = F.relu(out)
        out = self.conv1_2(out) # 222
        out = F.relu(out)
        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = F.relu(out)
        out = self.conv2_2(out) # 110
        out = F.relu(out)
        out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = F.relu(out)
        out = self.conv3_3(out) # 54
        out = F.relu(out)
        out = self.maxpool3(out) # 28
        
        out = self.conv4_1(out) # 26
        out = F.relu(out)
        out = self.conv4_2(out) # 26
        out = F.relu(out)
        out = self.conv4_3(out) # 26
        out = F.relu(out)
        out = self.maxpool4(out) # 14
        
        out = self.conv5_1(out) # 12
        out = F.relu(out)
        out = self.conv5_2(out) # 12
        out = F.relu(out)
        out = self.conv5_3(out) # 12
        out = F.relu(out)
        # out = self.maxpool5(out) # 7
        
        # # 展平
        # out = out.view(in_size, -1)
        
        # out = self.fc1(out)
        # out = F.relu(out)
        # out = self.fc2(out)
        # out = F.relu(out)
        # out = self.fc3(out)
        
        # out = F.log_softmax(out, dim=1)
        
        
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _load_pretrained_model(self):
        # pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        pretrain_dict = torch.load(r"modeling\backbone\vgg16_reducedfc.pth")
        
        
        state_dict = self.state_dict()
        print('pretrain:')
        for v,k in pretrain_dict.items():
            if 'weight' in v:
                print(v,k.shape)
        print('state:')
        for v,k in state_dict.items():
            if 'weight' in v:
                print(v,k.shape)

        model_dict = {}
        for k,v in pretrain_dict.items():
            if k.startswith('0.'):
                model_dict[k.replace('0', 'conv1_1')] = v
            elif k.startswith('2.'):
                model_dict[k.replace('2', 'conv1_2')] = v
            elif k.startswith('5.'):
                model_dict[k.replace('5', 'conv2_1')] = v
            elif k.startswith('7.'):
                model_dict[k.replace('7', 'conv2_2')] = v
            elif k.startswith('10.'):
                model_dict[k.replace('10', 'conv3_1')] = v
            elif k.startswith('12.'):
                model_dict[k.replace('12', 'conv3_2')] = v
            elif k.startswith('14.'):
                model_dict[k.replace('14', 'conv3_3')] = v
            elif k.startswith('17.'):
                model_dict[k.replace('17', 'conv4_1')] = v
            elif k.startswith('19.'):
                model_dict[k.replace('19', 'conv4_2')] = v
            elif k.startswith('21.'):
                model_dict[k.replace('21', 'conv4_3')] = v
            elif k.startswith('24.'):
                model_dict[k.replace('24', 'conv5_1')] = v
            elif k.startswith('26.'):
                model_dict[k.replace('26', 'conv5_2')] = v
            elif k.startswith('28.'):
                model_dict[k.replace('28', 'conv5_3')] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    import torch
    model = VGG16(pretrained=True)
    input = torch.rand(1, 3, 224, 224)
    output = model(input)
    print(output.size())
 

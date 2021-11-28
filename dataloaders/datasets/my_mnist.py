import os
import numpy as np
import scipy.misc as m

import sys
sys.path.append(r'D:\论文相关\pytorch-deeplab-xception-master')

from PIL import Image
import torch
from torch.utils import data
from mypath import Path
from torchvision import transforms

import torch.nn.functional as F

class RaiwaySegmentation(data.Dataset):
    NUM_CLASSES = 11

    def __init__(self, args, root=Path.db_root_dir('mnist'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.dataPath = os.path.join(self.root, 'mnist.npz') #train.txt

        mnistdata = np.load(self.dataPath)

        if(split == 'train'):
            self.x_data = mnistdata['x_train'][:10000]
            self.y_data = mnistdata['y_train'][:10000]
            
        elif(split == 'test'):
            self.x_data = mnistdata['x_test'][:5000]
            self.y_data = mnistdata['y_test'][:5000]
        elif(split == 'vla'):
            self.x_data = mnistdata['x_train'][10000:15000]
            self.y_data = mnistdata['y_train'][10000:15000]
        else:
            self.x_data = []
            self.y_data = []
        self.i_data = self.x_data


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):


        _img = Image.fromarray(self.x_data[index].astype('uint8')).convert('RGB')
        _img = _img.resize((224,224))
        n_img = np.array(_img)
    
        _target = np.zeros_like(n_img)# 224×224×3
        _target[n_img>125] = self.y_data[index]+1
        _target = _target[:,:,0]# 224×224

        img = torch.from_numpy(n_img).float()
        target = torch.from_numpy(_target).int()
        ins = torch.clamp(target,0,1)
        # print("mask max ",np.max(_target))
        # weight = torch.from_numpy(_weight).float()
        # weight = []
        w,h = img.shape[0],img.shape[1]

        
        _weight = torch.ones(1,h)
        weight = _weight.repeat(w,1)
   
        # weight = torch.rand(5)

        sample = {'image': img.permute([2,0,1]), 'label': target, 'instens': ins, 'weight':weight}
        return sample

    def mosaicCat(self):
        nx_data = self.x_data
        ny_data = self.x_data

        self.x_data = nx_data
        self.y_data = ny_data

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def my_loss(img, gt, wt):
    _img = img.permute([0,2,3,1])
    _gt = get_one_hot(gt,5)
    # _wt = wt.unsqueeze(3).repeat(1,1,1,5)
    _wt = wt
    print(_img)
    print(_gt)
    print(_wt)

    #  BEC 
    # t1 = _gt * torch.log(_img) + (1 - _gt)* torch.log(1-_img)
    # t2 = - _wt * t1
    # print(t2.mean())

    t1 = torch.exp(_img).sum(dim=3)
    t2 = (torch.exp(_img) * _gt).sum(dim=3)
    res = -torch.log(t1/t2)
    print(res.shape,_wt.shape)
    return (res*_wt).mean()

   
if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    root = Path.db_root_dir('mnist')
    mnistdata = np.load(os.path.join(root, 'mnist.npz'))
    print(mnistdata.files)
    x_test = mnistdata['x_test']
    y_test = mnistdata['y_test']
    x_train = mnistdata['x_train']
    x_train = mnistdata['x_train']
    print(x_test[1])

    _img = Image.fromarray(x_test[1].astype('uint8')).convert('RGB')
    _img = _img.resize((224,224))
    n_img = np.array(_img)
    
    n_target = np.zeros_like(n_img)
    n_target[n_img>125] = 255
    # _img.show()
    _target = Image.fromarray(n_target.astype('uint8'))
    _target.show()

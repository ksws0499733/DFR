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
    NUM_CLASSES = 4

    def __init__(self, args, root=Path.db_root_dir('my'), split="train",instanced = False, weighted = False):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        

        self.indexFilePath = os.path.join(self.root, self.split+'.txt') #train.txt
        file = open(self.indexFilePath,'r')
        if file is None:
            print("Cannot find %s file" %(split))
        list = file.readlines()#每一行数据写入到list中
        self.img_list = []
        self.cls_list = []
        self.ins_list = []
        self.weigth_list = []
        self.weighted = weighted
        self.instanced = instanced

        for str in list:
            substr = str.split()#默认移除所有空格
            print(substr)
            self.img_list.append(os.path.join(self.root,substr[0]))
            self.cls_list.append(os.path.join(self.root,substr[1]))
            if self.instanced:
                self.ins_list.append(os.path.join(self.root,substr[2]))
            if self.weighted:
                self.weigth_list.append(os.path.join(self.root,substr[3]))

        print("Found %d %s images" % (len(self.img_list), split))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        img_path = self.img_list[index]
        cls_path = self.cls_list[index]        

        _img = np.array(Image.open(img_path).convert('RGB'))    
        _target = np.array(Image.open(cls_path))             

        img = torch.from_numpy(_img).float()
        clss = torch.from_numpy(_target).int()
        
        if self.instanced:
            ins_path = self.ins_list[index]  
            _ins = np.array(Image.open(ins_path))   
            inss = torch.from_numpy(_ins).int()
        else:
            inss = None

        if self.weighted:
            weight_path = self.weigth_list[index]
            _weight = np.array(Image.open(weight_path))
            weight = torch.from_numpy(_weight).float()
        else:
            weight = None

        sample = {'image': img.permute([2,0,1]), 'label': clss, 'instens': inss, 'weight':weight}
        return sample

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

    Raiway_train = RaiwaySegmentation(args, split='train')

    dataloader = DataLoader(Raiway_train, batch_size=1, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        
        _img = sample['image']
        _gt = sample['label']
        _wt = sample['weight']

        # print(_wt.mean(dim=0))

        loss = F.cross_entropy(_img, 
                    _gt.long())
                
        print(loss)
        loss2 = my_loss(_img, 
                    _gt.long(), 
                    _wt)

        print(loss2)


        img = sample['image'].numpy()
        gt = sample['label'].numpy()
        wt = sample['weight'].numpy()
        for jj in range(sample["image"].size()[0]):

            print(ii,jj,img[jj].shape)
            print(ii,jj,gt[jj].shape)
            print(ii,jj,wt[jj].shape)
            print('-----')



            # tmp = np.array(img[jj])
            # # segmap = decode_segmap(tmp, dataset='cityscapes')
            # img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            # # img_tmp *= (0.229, 0.224, 0.225)
            # # img_tmp += (0.485, 0.456, 0.406)
            # img_tmp /= 255.0
            # segmap = np.array(gt[jj]).astype(np.uint8)
            # plt.figure()
            # plt.title('display')
            # plt.subplot(211)
            # plt.imshow(img_tmp)
            # plt.subplot(212)
            # plt.imshow(segmap)

        # if ii == 1:
        #     break

    plt.show(block=True)


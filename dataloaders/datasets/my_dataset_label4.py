import os
import numpy as np
import scipy.misc as m
from torchvision import transforms
from dataloaders import custom_transforms as tr

from PIL import Image,ImageDraw,ImageFont
import torch
from torch.utils import data
from mypath import Path
from torchvision import transforms

import torch.nn.functional as F
import json
import cv2.cv2 as cv2
import numpy.random as random
import math


def read_DataStru_json(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        strF = load_f.read()
        if len(strF) > 0:
            datas = json.loads(strF)
        else:
            datas = {}
    return datas


class RaiwaySegmentation(data.Dataset):
    NUM_CLASSES = 5

    def __init__(self, args, root=Path.db_root_dir('my'), split="train",instanced = False, weighted = False):
   
        root = Path.db_root_dir(args.dataset)
        print(root)
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.NUM_CLASSES = 5
        self.img_list = []
        self.cls_list = []


        self.indexPath = os.path.join(self.root, self.split) #/train/
        
        


        list = os.listdir(self.indexPath )
        
        for pth in list:
            if pth.endswith('.json') and split in ['train','val']:
                filename, _ = os.path.splitext(pth)
                    
                imgFile = os.path.join(self.indexPath,filename+'.jpg')
                clsFile = os.path.join(self.indexPath,filename+'_label.png')
                if os.path.isfile(imgFile) and os.path.isfile(clsFile):                    
                    self.img_list.append(imgFile)
                    self.cls_list.append(clsFile)
                for ii in ['_90','_180','_270']:
                    imgFile = os.path.join(self.indexPath,filename+ii+'.jpg')
                    clsFile = os.path.join(self.indexPath,filename+ii+'_label.jpg')
                    if os.path.isfile(imgFile) and os.path.isfile(clsFile):                    
                        self.img_list.append(imgFile)
                        self.cls_list.append(clsFile)
                    else:
                        print('error:= ',imgFile)
                        print('error:= ',clsFile)
                
            elif pth.endswith('.jpg') and self.split == 'test':
                imgFile = os.path.join(self.indexPath,pth)
                self.img_list.append(imgFile)
                self.cls_list.append(imgFile)
        # print('--debug:',len(list))
        # exit(-1)
        self.opp_list = []
        self.existz_list = []

        # if split in ['train','val']:
        #     self.indexPath = os.path.join(self.root, self.split+'_opp') #/train/
        #     print(self.indexPath)
        #     list = os.listdir(self.indexPath )
        #     for pth in list:  
        #         if pth.endswith('.jpg'):                     
        #             imgFile = os.path.join(self.indexPath,pth)
        #             if os.path.isfile(imgFile):
        #                 self.opp_list.append(imgFile)
        #                 self.img_list.append(imgFile)
        #                 self.cls_list.append('none')

        self.len1 = len(self.img_list)
        self.len2 = len(self.opp_list)


        self.weighted = weighted
        self.instanced = instanced
        self.img_size = 480
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        print("Found %d %s images, %d opps" % (len(self.img_list), split, len(self.opp_list)))

    def __len__(self):
        return len(self.img_list)
        # return len(self.img_list)+ len(self.opp_list)

    def __getitem__(self, index):


        if self.split == 'train' and not self.args.dataAug == 'no':
            _img, _target = self.data_augment(index)
            exist = np.array((1,))
        else:
            
            img_path = self.img_list[index]
            cls_path = self.cls_list[index]        
            exist = np.array((1,))
            _img = cv2.imread(img_path) #np.array(Image.open(img_path).convert('RGB'))
            if(os.path.isfile(cls_path)):
                _target = cv2.imread(cls_path) #np.array(Image.open(cls_path))  self.decodeJson(cls_path)
            else:
                _target = np.zeros_like(_img)

            _img, _target1 = self.exband(_img, _target)
            _target1 = _target1[:,:,0]
            _target2 = _target1.copy()
            
            _target1[_target1<2]= 0
            _target1[_target1>1]= 1
            _target2[_target2==2]= 1
            _target2[_target2==3]= 2
            _target2[_target2>=4]= 3  
                
        # print('===',_target1.shape)
        # print('===',_target2.shape)

        #print(np.max(_target),np.min(_target))
        _ins = None
        _wt = None

        img = torch.from_numpy(_img).float()
        clss1 = torch.from_numpy(_target1).int()
        clss2 = torch.from_numpy(_target2).int()
        
        
        if self.instanced: 
            inss = torch.from_numpy(_ins).int()
        else:
            inss = None

        if self.weighted:
            weight = torch.from_numpy(_wt).float()
        else:
            weight = None

        sample = {'image': img, 'label1': clss1, 'label2': clss2, 'exist':  exist   ,'instens': inss, 'weight':weight}
        return self.transform_val(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
        tr.ToTensor()])
        return composed_transforms(sample)

    def exband(self, img, gt, shape=(720,576)):
        if img is None:
            img2 = np.zeros((720,576,3),dtype='uint8')
            gt2 = np.zeros((720,576,3),dtype='uint8')

        else:
            img2 = cv2.resize(img,shape)
            gt2 = cv2.resize(gt, shape)

        # img2 = np.zeros((shape[0],shape[1],3))
        # x1 = int(288-img.shape[0]/2)
        # x2 = x1 + img.shape[0]
        # y1 = int(360-img.shape[1]/2)
        # y2 = y1 + img.shape[1]
        # img2[x1:x2,y1:y2,:] = img
        # gt2 = np.zeros((shape[0],shape[1]))
        # gt2[x1:x2,y1:y2] = gt

        # print(img2.shape)
        # print(gt2.shape)
        return img2, gt2
    
    def data_dig(self, img, mH = 100, mW = 100):
        sx,sy = random.randint(0,mW),random.randint(0,mH)
        eRatio = random.uniform(0.1,0.6)
        h, w = img.shape[:2]
        para = random.rand()
        if para<0.25:
            color = 0
        elif para < 0.5:
            color = 1
        else:
            color = random.rand()
        for ii in range(w//mW):
            for jj in range(h//mH):
                # x1a = min(sx + ii*mW, w-1)
                # y1a = min(sy + jj*mH, h-1)
                # x2a = min(sx + ii*mW + mW*eRatio , w-1)
                # y2a = min(sy + jj*mH + mH*eRatio , h-1)
                x1a,y1a,x2a,y2a = min(sx + ii*mW, w-1),min(sy + jj*mH, h-1),min(sx + ii*mW + int(mW*eRatio) , w-1),min(sy + jj*mH + int(mH*eRatio) , h-1)
                
                img[x1a:x2a,y1a:y2a] = color
        return img

    def data_augment(self,index):
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [random.randint(0, len(self.img_list) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img_path = self.img_list[index]
            cls_path = self.cls_list[index]        
            _img = cv2.imread(img_path) #np.array(Image.open(img_path).convert('RGB'))
            # _target = cv2.imread(cls_path) #np.array(Image.open(cls_path))  self.decodeJson(cls_path)
            if(os.path.isfile(cls_path)):
                _target = cv2.imread(cls_path) #np.array(Image.open(cls_path))  self.decodeJson(cls_path)
            else:
                _target = np.zeros_like(_img)
            # _target[_target<=2]= 0
            # _target[_target==3]= 1
            # _target[_target==4]= 2
            _img, _target = self.exband(_img, _target)
            _target =  np.clip(_target,0,self.NUM_CLASSES-1)
            if self.args.dataAug in ['all','affine']: 
                _img,_target = random_affine(_img,_target)

            h, w = _img.shape[:2]
            # print(h,w)
            #_img = _img.astype(np.float32)
            #_img = cv2.normalize(_img,_img)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, _img.shape[2]), 127, dtype=np.uint8)  # base image with 4 tiles
                tgt4 = np.full((s * 2, s * 2, _target.shape[2]), 255, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = _img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            tgt4[y1a:y2a, x1a:x2a] = _target[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        if self.args.dataAug in ['all','dig']:        
            img4 = self.data_dig(img4)
        return img4,tgt4[:,:,0]

def random_affine(img, targets, degrees=10, translate=.1, scale=.6, shear=10, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        targets = cv2.warpAffine(targets, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR,borderValue= 0)

    return img, targets


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


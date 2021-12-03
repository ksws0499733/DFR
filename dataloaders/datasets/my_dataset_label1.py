import os
import sys

sys.path.append(r'/home/user1106/DeepNets/Deeplab-upload2')
print(sys.path)

import numpy as np
from torchvision import transforms
from dataloaders import custom_transforms as tr

import torch
from torch.utils import data
from mypath import Path
from torchvision import transforms

import torch.nn.functional as F
import json
import cv2.cv2 as cv2
import numpy.random as random
import math


class RaiwaySegmentation(data.Dataset):
    NUM_CLASSES = 4

    def __init__(self, args, root=None, split="train"):
        if root is None:
            root = Path.db_root_dir(args.dataset)
        print(root)
        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.NUM_CLASSES = 4

        self.img_list = []
        self.cls_list = []

        total_len = -1
        posi_len = -1
        nega_len = -1


        if split in ['train','val']:
            indexPath1 = os.path.join(self.root, self.split) #/positive/ 
            indexPath2 = os.path.join(self.root, self.split+'_negative') #/negative/           
            # print(indexPath1)
            nega_len = self.list_negative(indexPath2)
            total_len = self.list_positive(indexPath1)
            

        elif self.split == 'test':  
            indexPath3 = os.path.join(self.root, self.split) #/test/    
            total_len = self.list_negative(indexPath3)        

        self.img_size = 480
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        print("Found %d %s images, %d opps" % (len(self.img_list), split, nega_len))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        _img, _target = self.data_read(index)

        if self.split == 'train' and not self.args.dataAug == 'no':
            _img, _target = self.data_augment(_img, _target)   

        img = torch.from_numpy(_img).float()
        clss = torch.from_numpy(_target).int()   
        sample = {'image': img, 'label': clss}

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
    
    def data_augment(self,img, target):
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        
        h, w = img.shape[:2]
        img4 = np.full((s * 2, s * 2, img.shape[2]), 127, dtype=np.uint8)  # base image with 4 tiles
        tgt4 = np.full((s * 2, s * 2), 0, dtype=np.uint8)  # base image with 4 tiles


        indices = [-1] + [random.randint(0, len(self.img_list) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):

            if i == 0:
                _img, _target = img, target
            else:
                _img, _target = self.data_read(index)

            if self.args.dataAug in ['all','affine']: 
                _img,_target = random_affine(_img,_target)

            # if self.args.dataAug in ['all','shadow']: 
            #     _img = random_shadow(_img)

            if self.args.dataAug in ['all','dig']: 
                _img = random_dig(_img)            
 
            if i == 0:  # top left
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

        return img4,tgt4

    def data_read(self, index):
        img_path = self.img_list[index]
        cls_path = self.cls_list[index]   

        _img = cv2.imread(img_path) 
        if(os.path.isfile(cls_path)):
            _target = cv2.imread(cls_path)
        else:
            _target = np.zeros_like(_img)

        _img, _target = self.exband(_img, _target)    
        _target =  np.clip(_target[:,:,0],0,self.NUM_CLASSES-1)
        return _img, _target

    def list_positive(self, indexPath):
        list = os.listdir(indexPath )
        for pth in list:
            if pth.endswith('.jpg'):   
                filename, _ = os.path.splitext(pth)                    
                imgFile = os.path.join(indexPath,pth)
                clsFile = os.path.join(indexPath,filename+'_label.png')
                if os.path.isfile(imgFile) and os.path.isfile(clsFile):
                    self.img_list.append(imgFile)
                    self.cls_list.append(clsFile)
        return len(self.img_list)

    def list_negative(self, indexPath):
        list = os.listdir(indexPath )
        for pth in list:  
            if pth.endswith('.jpg'):                     
                imgFile = os.path.join(indexPath,pth)
                if os.path.isfile(imgFile):
                    self.img_list.append(imgFile)
                    self.cls_list.append('none')  
        return len(self.img_list)

def random_dig(img, mH = 100, mW = 100):
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
                x1a,y1a,x2a,y2a = min(sx + ii*mW, w-1),min(sy + jj*mH, h-1),min(sx + ii*mW + int(mW*eRatio) , w-1),min(sy + jj*mH + int(mH*eRatio) , h-1)
                
                img[x1a:x2a,y1a:y2a] = color
        return img


def random_affine(img, targets, degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
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

def random_shadow(img):

    height = img.shape[0]
    width = img.shape[1]

    mask = np.zeros_like(img)
    p1x, p1y = int(random.uniform(0, width)), int(random.uniform(0, height))
    p2x, p2y = int(random.uniform(0, width)), int(random.uniform(0, height))

    dx, dy = p2x - p1x + 1e-5, p2y - p1y + 1e-5
    k1 = min(max(np.array([p1x / dx, (p1x-width)/dx, p1y / dy, (p1y-height)/dy]), 0))
    k2 = max(min(np.array([-p1x / dx, -(p1x-width)/dx, -p1y / dy, -(p1y-height)/dy]), 0))

    x1, y1 = p1x - k1*dx, p1y - k1*dy
    x2, y2 = p2x + k2*dx, p2y + k2*dy
    bd_list = [(x1,y1),(x2,y2)]

    D=[(0,0),(width,0), (0,height), (0,0)]
    for d in D:
        if -dy*d[0] + dx*d[1] > 0:
            bd_list.append(d)

    cv2.fillConvexPoly(mask, np.array(bd_list), (255,255,255))

    img = img * (mask.astype("float32")/510 + 0.5)

    return img.astype('unint8')

def railtrack_shadow(img, target):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(target)
    mask[target == 2] = 1
    mask0 = mask.copy()

    a = np.random.randint(-64,65)
    for i in range(1,9):
        
        shift = 2**i * a/abs(a)
        if abs(shift) < abs(a):
            mask = mask + in_railtrack_shadow(mask, shift)
        else:
            mask = mask + in_railtrack_shadow(a)
            break

    


    return img

def in_railtrack_shadow(img, shift):
    width = img.shape[1]
    out = np.zeros_like(img)
    if shift > 0:
        out[:,0:width-shift] = img[:,shift:width]
    else:
        out[:,shift:width] = img[:,0:width-shift]
    return out

   
if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # args.base_size = 513
    # args.crop_size = 513
    args.dataAug = 'all'

    Raiway_train = RaiwaySegmentation(args,
                            root= r'/home/user1106/DataSet/iRailway' 
                            ,split='train')
    dataloader = DataLoader(Raiway_train, batch_size=1, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        
        img = sample['image'].numpy()
        gt = sample['label'].numpy()

        for jj in range(sample["image"].size()[0]):

            print(ii,jj,img[jj].shape)
            print(ii,jj,gt[jj].shape)
            print('-----')

            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp /= 255.0
            segmap = np.array(gt[jj]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 2:
            break
    plt.show(block=True)


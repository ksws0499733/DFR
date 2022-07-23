# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
from pathlib import Path
import os
import numpy as np
from torchvision import transforms
from dataloaders import custom_transforms as tr, utils
import torch
from torch.utils import data
import dataloaders
from mypath import Path
import torch.nn.functional as F
import json
import cv2.cv2 as cv2
import torch

class iRailway(data.Dataset):
    def __init__(self, img_folder, ann_folder=None, 
                        label_folder=None, 
                        transforms=None, 
                        return_masks=True, 
                        split='train',
                        repeat = 10,
                        isOnline = True):

        if not isinstance(img_folder, list):
            img_folder = [img_folder,]

        self.split = split
        self.isOnline = isOnline

        self.img_list = []
        self.ann_list = []
        self.cls_list = []
        self.ins_list = []
        self.mask_list = []
        print('    ',split)
        for img_fd in img_folder:
            self.load_from_subdir(img_fd, split, 
                                            repeat,
                                            ann_folder,
                                            label_folder)

            print('\t',img_fd,' : ', len(self.img_list))

        self.transforms = transforms
        self.return_masks = return_masks
        self.len = len(self.img_list)

    def load_from_subdir(self, img_folder,
                                split,
                                repeat,
                                ann_folder=None,
                                label_folder=None):
        img_list = os.listdir(img_folder)
        if split == 'test':
            for imgfile in img_list:
                if imgfile.endswith('.jpg'):
                    clsfile = imgfile.replace('.jpg','_cls.png')

                    self.img_list.append(os.path.join(img_folder,imgfile))
                    self.cls_list.append(os.path.join(img_folder,clsfile))

        elif split in ['train','var','save']:

            if ann_folder is None:
                ann_folder = os.path.join(img_folder,"Info")

            if not os.path.isdir(ann_folder):
                return
            
            if label_folder is None:
                label_folder = os.path.join(ann_folder,"class_mask_png")        
            ann_list = os.listdir(ann_folder)
            label_list = os.listdir(label_folder)

            for annfile in ann_list:
                if annfile.endswith('.json'):
                    imgfile = annfile.replace('.json', '.jpg')
                    clsfile = annfile.replace('.json', '_cls.png')
                    insfile = annfile.replace('.json', '_ins.png')
                    if imgfile in img_list and clsfile in label_list and insfile in label_list:
                        for i in range(repeat):                    
                            self.img_list.append(os.path.join(img_folder,imgfile))
                            self.cls_list.append(os.path.join(label_folder,clsfile))
                            self.ins_list.append(os.path.join(label_folder,insfile))

                            with open(os.path.join(ann_folder,annfile), 'r') as f:
                                ann_info = json.load(f)
                                self.ann_list.append(ann_info)

    def __getitem__(self, idx):
        if self.isOnline:
            return self.__getitem__online(idx)
        else:
            return self.__getitem_offline(idx)

    def __getitem_offline(self, idx):


        if self.split in ['train','var','save']:

            img_path = self.img_list[idx]        
            ins_path = self.ins_list[idx]
            cls_path = self.cls_list[idx]

            ann_info = self.ann_list[idx]


            _img = cv2.imread(img_path)
            if _img is None:
                return self.image_error(idx)


            if(os.path.isfile(cls_path)):
                _cls = cv2.imread(cls_path)
            else:
                _cls = np.zeros_like(_img)
            _cls = np.clip(_cls, 0, 3)
            _cls = _cls.astype(np.float32)

            ids = np.array([0] + [ann['ins_id'] for ann in ann_info['shapes']])

            _ins = cv2.imread(ins_path)
            _ins = rgb2id(_ins)#颜色 = ID        
            _ins = _ins == ids[:, None, None]
            _ins = _ins.astype(np.uint8)

            img = _img
            clss = _cls[:,:,0]
            inss = _ins
            mask = (clss > 1).astype(np.float32)
            sample = {'image': img, 'label': clss,'instence': inss,'mask':mask}

        elif self.split == 'test':
            img_path = self.img_list[idx]        

            _img = cv2.imread(img_path)
            if _img is None:
                return self.image_error(idx)

            _cls = np.zeros_like(_img).astype(np.float32)
            _mask = np.zeros_like(_img).astype(np.float32)

            img = _img
            clss = _cls[:,:,0]
            inss = _img.copy().transpose((2, 0, 1))
            mask = _mask[:,:,0]

            sample = {'image': img, 'label': clss,'instence': inss,'mask':mask}
            return self.transform_val(sample)

        if self.transforms is None:
            return self.transform_val(sample)
        else:
            return self.transforms(sample)

    def __getitem__online(self, idx):

        sample_list = []
        for _idx in range(idx, idx+8):
            _idx = min(self.len -1, _idx)
            # print('----------',_idx)

            if self.split in ['train','var','save']:

                img_path = self.img_list[_idx]        
                ins_path = self.ins_list[_idx]
                cls_path = self.cls_list[_idx]

                ann_info = self.ann_list[_idx]


                _img = cv2.imread(img_path)
                if _img is None:
                    return self.image_error(_idx)


                if(os.path.isfile(cls_path)):
                    _cls = cv2.imread(cls_path)
                else:
                    _cls = np.zeros_like(_img)
                _cls = np.clip(_cls, 0, 3)
                _cls = _cls.astype(np.float32)

                ids = np.array([0] + [ann['ins_id'] for ann in ann_info['shapes']])

                _ins = cv2.imread(ins_path)
                _ins = rgb2id(_ins)#颜色 = ID        
                _ins = _ins == ids[:, None, None]
                _ins = _ins.astype(np.uint8)

                img = _img
                clss = _cls[:,:,0]
                inss = _ins
                mask = (clss > 1).astype(np.float32)
                sample = {'image': img, 'label': clss,'instence': inss,'mask':mask}
                
            elif self.split == 'test':
                img_path = self.img_list[_idx]        

                _img = cv2.imread(img_path)
                if _img is None:
                    return self.image_error(_idx)

                _cls = np.zeros_like(_img).astype(np.float32)
                _mask = np.zeros_like(_img).astype(np.float32)

                img = _img
                clss = _cls[:,:,0]
                inss = _img.copy().transpose((2, 0, 1))
                mask = _mask[:,:,0]

                sample = {'image': img, 'label': clss,'instence': inss,'mask':mask}

            if self.transforms is None or self.split == 'test':
                sample = self.transform_val(sample)
            else:
                sample = self.transforms(sample)
        
            sample_list.append(sample)
        return sample_list


    def image_error(self, idx = 0):

        print("Image idx:{} is error image; path:{}".format(idx, self.img_list[idx]))
        sample = {'image': None, 'label': None,'instence': None,'mask':None}
        if self.transforms is None:
            return self.transform_val(sample)
        else:
            return self.transforms(sample)

    def transform_val(self, sample):
        composed_transforms = tr.Compose([
            tr.crop_auto(0,0,512,512),
            # tr.resize(800,512),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __len__(self):
        return len(self.img_list)

    def get_height_and_width(self, idx):
        ann_info = self.ann_list[idx]
        height = ann_info['imageHeight']
        width = ann_info['imageWidth']
        return height, width


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 2] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 0]
    return int(color[1] + 256 * color[1] + 256 * 256 * color[0])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color

import json
def build_base(state, args):
    dataset_root0 = Path.db_root_dir(args.dataset)

    dataset_file = os.path.join(dataset_root0, 'dataset_used.txt')

    with open(dataset_file,'r') as f:
        dataset_root_list = f.readlines()

    dataset_list = []

    for dataset_r in dataset_root_list:
        pth = os.path.join(dataset_root0,dataset_r[:-1])# remove '\n' in the end
        if os.path.isdir(pth):
            dataset_list.append(pth)  

    PATHS = {
        "train": dataset_list,
        "val":  dataset_list,
        "test": dataset_list,
        "save": dataset_list
    }
    return PATHS

def build_offline(state, args):

    PATHS = build_base(state, args)

    if args.dataAug == 'no':
        transforms=tr.Compose([
                            tr.resize(800,512),
                            tr.MaskAffine(),
                            tr.ToTensor()
                           ])
    else:
        transforms=tr.Compose([
                            tr.crop_auto(0,0,512,512),
                            tr.RandomAffine(),
                            tr.MaskAffine(),
                            tr.ToTensor()
                           ])
    
    img_folder = PATHS[state]
    dataset = iRailway(img_folder,
                        split=state,
                        transforms=transforms,
                        repeat= args.dataRepeat,
                        isOnline=False
                        )


    return dataset

def build_online(state, args):

    PATHS = build_base(state, args)

    if args.dataAug == 'no':
        transforms=tr.Compose([
                            tr.resize(800,512),
                            tr.ToTensor()
                           ])
    else:
        transforms=tr.Compose([
                            tr.crop_auto(0,0,512,512),
                            # tr.RandomAffine(),
                            tr.MaskAffine(),
                            tr.ToTensor()
                           ])
    
    img_folder = PATHS[state]
    dataset = iRailway(img_folder,
                        split=state,
                        transforms=transforms,
                        repeat= args.dataRepeat,
                        isOnline=True
                        )
    return dataset

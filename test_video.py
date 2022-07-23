import argparse
from ast import While
from genericpath import isfile
import os
import numpy as np
import cv2.cv2 as cv2
from tqdm import tqdm
import blockline
import torch
import sys
# sys.path.append(r'/home/user1106/DeepNets/Deeplab-pytorch')
print(sys.path)

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.RailTrack import RTrackNet

def saveimage(img,pred):
    N,H,W,C = img.shape
    cls_color = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,0,255]]

    
    _img = img[0].astype('uint8')
    _prd = pred[0].astype('float32')
    cls_mask = _prd*30
    im = np.zeros_like(_img)
    # im = Image.fromarray(cls_mask).convert('RGB')
    im_R = np.zeros_like(_prd)
    im_G = np.zeros_like(_prd)
    im_B = np.zeros_like(_prd)
    for j in range(5):
        im_R[_prd > j-0.1 ] = cls_color[j][0]
        im_G[_prd > j-0.1 ] = cls_color[j][1]
        im_B[_prd > j-0.1 ] = cls_color[j][2]
    im[:,:,0] = im_R
    im[:,:,1] = im_G
    im[:,:,2] = im_B
    imgout = cv2.addWeighted(im,0.5,_img,0.5,0)



    segmentor = blockline.railSegmenter()
    _pred = pred
    _pred = _pred.astype('uint8')
    _pred[_pred<2] = 0
    _pred[_pred>=2] = 255


    mask_batch, draw_batch, block_batch, bd_batch = segmentor.run2(img, pred.astype('uint8'), 255)#, img)

    return imgout, mask_batch, draw_batch, block_batch, bd_batch



def test(model, videoFile, first_mask=None, outroot='',outputFile='out.mp4'):
    print(outputFile)

    model.eval()
    outfile = os.path.join(outroot,outputFile)

    vcap = cv2.VideoCapture(videoFile)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vwrt = cv2.VideoWriter(outfile, fourcc, 20.0, (720,576) )
    if not vcap.isOpened():
        print("cannot open camera")
        exit()
    if isinstance(first_mask, str):
        if isfile(first_mask):
            _mask = cv2.imread(first_mask)
            _mask = cv2.resize(_mask,(720,576))
            _mask = (_mask[:,:,0] > 1).astype(np.float32)
        else:
            _mask = np.zeros((720,576))
    else:
        _mask = first_mask  # h*w

    mask = torch.from_numpy(_mask).unsqueeze(0).unsqueeze(0) # 1*1*h*w
    
    cnt = 0
    while True:
        vcap.set(cv2.CAP_PROP_POS_FRAMES, cnt)
        ret, _img = vcap.read()

        if not ret:
            break

        _img = cv2.resize(_img,(720,576))

        image = torch.from_numpy(_img.astype(np.float32)).unsqueeze(0)  # 1*h*w*3
        image = image.permute(0,3,1,2)               # 1*3*h*w
        

        image, mask = image.cuda(), mask.cuda()

        print(cnt,'------', mask.shape, image.shape)
        cnt += 10

        input = torch.cat((image,mask), dim= 1)

        with torch.no_grad():
            output = model(input)
        
        mask = torch.argmax(output,dim=1) # 1*h*w
        a = torch.ones_like(mask)
        b = torch.zeros_like(mask)
        mask = torch.where(mask>1,a,b).unsqueeze(0)# 1*1*h*w


        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        
        image = image.cpu().permute(0,2,3,1).numpy()

        imgout, mask_batch, draw_batch, block_batch, bd_batch = saveimage( image,pred)
        cv2.imshow('imgout',imgout)
        cv2.waitKey(1)

        vwrt.write(imgout)
    
    vcap.release()
    vwrt.release()




def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    parser.add_argument('--dataset', type=str, default='iRailway.mp4',
                        help='dataset file')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='model path') 
    parser.add_argument('--output', type=str, default='out.mp4',
                        help='output file')
    parser.add_argument('--fisrtMask', type=str, default='mask.png',
                        help='output file')


    args = parser.parse_args()  

    model = RTrackNet(backbone='CSPnet',
                            neck='A-S',
                            output_stride=16, 
                            inChannal= 4,
                            num_classes= 4)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    if args.model is not None:
        #9.1--- load checkpoint
        if not os.path.isfile(args.model):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.model))
        checkpoint = torch.load(args.model)

        model.load_state_dict(checkpoint['state_dict'])

    test(model, args.dataset, 
                first_mask=args.fisrtMask, 
                outroot='run',
                outputFile=args.output)
    
# python test_video.py --dataset=/home/user1106/DataSet/mananxian.mp4 \
# --model=run/iRailway/CSPnet-A-S-ce-iRailway/experiment_7/best_model.pth\
# --fisrtMask=/home/user1106/DataSet/all_dataset2/nanjing01/Info/class_mask_png/dji000009_cls.png



if __name__ == "__main__":

    
    main()


import argparse
import os
import numpy as np
import torch

from PIL import Image
import cv2.cv2 as cv2

def saveimage(img,pred, root, startID):
    N,H,W,C = img.shape[0],img.shape[1],img.shape[2],img.shape[3]
    cls_color = [[0,0,0],[255,0,0],[0,255,0],[0,0,255]]
    if not os.path.isdir(root):
        os.makedirs(root)
    for i in range(N):
        _img = img[i].astype('uint8')
        _prd = pred[i].astype('float32')
        cls_mask = _prd*30
        im = np.zeros_like(_img)
        # im = Image.fromarray(cls_mask).convert('RGB')
        im_R = np.zeros_like(_prd)
        im_G = np.zeros_like(_prd)
        im_B = np.zeros_like(_prd)
        for j in range(4):
            im_R[_prd > j-0.1 ] = cls_color[j][0]
            im_G[_prd > j-0.1 ] = cls_color[j][1]
            im_B[_prd > j-0.1 ] = cls_color[j][2]
        im[:,:,0] = im_R
        im[:,:,1] = im_G
        im[:,:,2] = im_B
        im = Image.fromarray(im)
        imbase = Image.fromarray(_img)
        imout = Image.blend(imbase,im, 0.5)
        npth = os.path.join(root,'{:0>6}.jpg'.format(startID+i))
        imout.save(npth)


    print(img.shape)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    
    parser.add_argument('--model_path', type=str, default='net1_mobile_deeplab.pkl',                            
                            help='model name')

    parser.add_argument('--img_path', type=str, default=None,                        
                        help='dataset name')
    parser.add_argument('--output_name', type=str, default="best_model",                        
                        help='dataset name')
    parser.add_argument('--cpu', action='store_true', default=False,                        
                        help='is use cuda')
    
    args = parser.parse_args()

    if args.img_path is None:
        img_path = r'data_set/test'
    else:
        img_path = args.img_path
    
    if not os.path.isfile(args.model_path):
        print('========> no such file')
        return
    outName = args.output_name
    model = torch.load(args.model_path) 
    model.eval()

    img_list = os.listdir(img_path)
    root = os.path.join(args.model_path[:-23],'test_output')
    if not os.path.isdir(root):
        os.mkdir(root)
    for pth in img_list:
        imPth = os.path.join(img_path,pth)
        print(imPth)
        _img = np.array(Image.open(imPth).convert('RGB'))
        _img = cv2.resize(_img, (640,480))

        input = torch.from_numpy(_img).float().permute(2,0,1).unsqueeze(0)
        input = input.cuda()
        print(input.shape)

        output = model(input)      #网络模型输出

        pred = output.data.cpu().squeeze().numpy()  #输出放到CPU中，以numpy数据格式

        
        pred = np.argmax(pred, axis=0)      #输出中取最大值的序号作为分类号（形成图片
        #print(pred.shape)

        cls_color = [(0,0,0),(255,0,0),(0,255,0),(0,0,255)]
        
        _img = _img.astype('uint8')
        _prd = pred.astype('float32')

        im = np.zeros_like(_img)
        im_R = np.zeros_like(_prd)
        im_G = np.zeros_like(_prd)
        im_B = np.zeros_like(_prd)
        for j in range(4):
            im_R[_prd > j-0.1 ] = cls_color[j][0]
            im_G[_prd > j-0.1 ] = cls_color[j][1]
            im_B[_prd > j-0.1 ] = cls_color[j][2]
        im[:,:,0] = im_R
        im[:,:,1] = im_G
        im[:,:,2] = im_B
        im = Image.fromarray(im)
        imbase = Image.fromarray(_img)
        imout = Image.blend(imbase,im, 0.5)
        npth = os.path.join(root,pth)
        imout.save(npth)


if __name__ == "__main__":

    
    main()

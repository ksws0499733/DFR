
import os
import numpy as np
import cv2.cv2 as cv2
import blockline



def saveimage(img,pred, root,subroot='0', startID=0, video_writer_list = None):
    N,H,W,C = img.shape
    cls_color = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,0,255]]

    root = os.path.join(root,subroot)
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
        for j in range(5):
            im_R[_prd > j-0.1 ] = cls_color[j][0]
            im_G[_prd > j-0.1 ] = cls_color[j][1]
            im_B[_prd > j-0.1 ] = cls_color[j][2]
        im[:,:,0] = im_R
        im[:,:,1] = im_G
        im[:,:,2] = im_B
        imgout = cv2.addWeighted(im,0.5,_img,0.5,0)
        if video_writer_list is None:
            npth = os.path.join(root,'{:0>6}.jpg'.format(startID+i))
            # print('cur save: ',npth)
            npth2 = os.path.join(root,'{:0>6}_color.jpg'.format(startID+i))            
            cv2.imwrite(npth,imgout)
            cv2.imwrite(npth2,im)
        else:
            video_writer_list[0].write(imgout)


    segmentor = blockline.railSegmenter()
    _pred = pred
    _pred = _pred.astype('uint8')
    _pred[_pred<2] = 0
    _pred[_pred>=2] = 255
    
    for i in range(N):
        if video_writer_list is None:
            _prd = _pred[i].astype('float32')
            npth2 = os.path.join(root,'{:0>6}_rail.jpg'.format(startID+i))
            cv2.imwrite(npth2,_prd)

    mask_batch, draw_batch, block_batch, bd_batch = segmentor.run2(img, pred.astype('uint8'), 255)#, img)
    for i, (maks, draw, block, bd) in enumerate( zip(mask_batch,draw_batch,block_batch,bd_batch)):
        drawName =  os.path.join(root,'{:0>6}_draw0.jpg'.format(startID+i))
        maskName = os.path.join(root,'{:0>6}_railmask.jpg'.format(startID+i))
        bolckName = os.path.join(root,'{:0>6}_block.jpg'.format(startID+i))
        bdkName = os.path.join(root,'{:0>6}_bd.jpg'.format(startID+i))
        cv2.imwrite( drawName,draw)
        cv2.imwrite( maskName,maks)
        cv2.imwrite( bolckName,block)
        cv2.imwrite( bdkName,bd)


    # print(img.shape)

    
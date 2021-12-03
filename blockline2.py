# -*- coding: utf-8 -*
import cv2.cv2 as cv2
import numpy as np
from functools import cmp_to_key
import os

from numpy.core.fromnumeric import argmin

# global gnumber = 0

class railSegmenter:
    def __init__(self):
        self.rlines = []
        self.gnumber = 0

    def init_batch(self, label_batch, canvas_batch):
        if len(label_batch.shape) == 3:  # N,W,H
            N,H,W = label_batch.shape
        elif len(label_batch.shape) == 4: #N,C,W,H
            N,C,H,W = label_batch.shape
        else:
            print('input data has no enough dims')
            return None, None, None, None
        mask_batch = np.zeros((N,H,W))
        block_batch = np.zeros((N,H,W,3))
        bdmap_batch = np.zeros((N,H,W,3))
        if canvas_batch is None:
            print('0',label_batch.shape)
            canvas_batch = np.zeros((N,H,W,3), dtype = 'uint8')+220
        return mask_batch,block_batch,bdmap_batch,canvas_batch


    def run(self,img_batch, label_batch, color = 1, blockLines_batch=None, is_draw = True):
        
        # print('img_batch',img_batch.shape)
        # print('label_batch',label_batch.shape)
        
        maskLabel_batch,blockPoints_batch,maskImage_batch,blockLines_batch = self.init_batch(label_batch, blockLines_batch)

        for id, (img, lable) in enumerate(zip(img_batch, label_batch)):
            print('current id:',self.gnumber)
            self.gnumber +=1
            if len(lable.shape) == 2:
                lable0 = np.zeros_like(lable)
                lable0[lable>=1] = 255
                lable1 = np.zeros_like(lable)
                lable1[lable>=2] = 255            
                lable2 = np.zeros_like(lable)
                lable2[lable>=3] = 255

            else:
                lable0 =lable[0]
                lable1 =lable[1]
                lable2 =lable[2]

            bPoint_list = self.find_blockPoints(lable0, 16) #ballast            
            rPoint_list = self.find_blockPoints(lable1, 16) #railway
            tPoint_list = self.find_blockPoints(lable2, 16) #track

            for bp in rPoint_list:
                bp.check(lable2,lable0)
                bp.draw(blockPoints_batch[id], lcolor=(0,0,255))     
                # cv2.imshow('block_batch', block_batch[id])
                # cv2.waitKey()     

            # self.blines = self.find_blockLines(bPoint_list)
            self.rlines = self.find_blockLines(rPoint_list)
            # self.tlines = self.find_blockLines(tPoint_list)

            for line in self.rlines:                  
                line.draw_raw(blockLines_batch[id])
                line.check(lable2, lable0)   

            self.draw(blockLines_batch[id]) 
            self.full(maskLabel_batch[id], color)
            maskImage_batch[id,:,:,0] = (img[:,:,0] + maskLabel_batch[id])/2
            maskImage_batch[id,:,:,1] = (img[:,:,1] + maskLabel_batch[id])/2
            maskImage_batch[id,:,:,2] = (img[:,:,2] + maskLabel_batch[id])/2
            
        return maskLabel_batch,blockLines_batch,blockPoints_batch,maskImage_batch

    def find_blockPoints(self, label, bin_number=20, axis=0):
        W = label.shape[1]
        # print("W",W)
        imgs = np.split(label, bin_number, axis= axis)
        bPoint_list = []
        for idx0, sub_img in enumerate(imgs):
            h,w = sub_img.shape
            # print(w,h)
            contours, _ = cv2.findContours(sub_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    bLine = blockPoint(idx0, h, contour, W)
                    bPoint_list.append(bLine)
        return bPoint_list

    def find_blockLines(self, blockPoint_list):
        NN = len(blockPoint_list)
        corMat = np.zeros((NN,NN))

        for id1, bp1 in enumerate(blockPoint_list) :
            for id2, bp2 in enumerate(blockPoint_list):  
                corMat[id1, id2] = bp1.corr(bp2, id2)      

        head_list = []
        corMat_head = np.sum(corMat, axis=0)
        head_list = np.where(corMat_head==0)[0].tolist()
        seed_list = [[x,] for x in head_list]

        corMat[corMat==0] = 1e5

        while 1:            
            min_cor = 1e5
            for hi, head in enumerate(head_list) :
                mcor = np.min(corMat[head])
                if mcor < min_cor:
                    cdi = hi
                    cdj = np.argmin(corMat[head])
                    min_cor = mcor

            if min_cor < 1e5:
                seed_list[cdi].append(cdj)
                head_list[cdi] = cdj
                corMat[:,cdj] = 1e5
            else:
                break

        lines = []
        for line in seed_list:
            newList = []
            for idx in line:
                newList.append(blockPoint_list[idx])
            lines.append(blockLine(newList))
        return lines
    
    def run4train(self,label_batch, color = 1):
        if len(label_batch.shape) == 3:
            mask_batch = np.zeros((label_batch.shape[0],label_batch.shape[1],label_batch.shape[2]))
        elif len(label_batch.shape) == 4:
            mask_batch = np.zeros((label_batch.shape[0],label_batch.shape[2],label_batch.shape[3]))
        else:
            print('input data has no enough dims')
            return None


        for id, lable in enumerate(label_batch):
            # print(len(lable.shape))
            if len(lable.shape) == 2:
                lable0 = np.zeros_like(lable)
                lable0[lable>=1] = 255

                lable1 = np.zeros_like(lable)
                lable1[lable>=2] = 255
            
                lable2 = np.zeros_like(lable)
                lable2[lable>=3] = 255

            else:
                lable0 =lable[0]
                lable1 =lable[1]
                lable2 =lable[2]

            rPoint_list = self.find_blockPoints(lable1, 16) #railway
            for bp in rPoint_list:
                bp.check(lable2,lable0)    

            self.rlines = self.find_blockLines(rPoint_list)

            for line in self.rlines:                  
                line.check(lable2, lable0)  

            self.full(mask_batch[id,:,:], color)
            
        return mask_batch

    def draw(self, canvas, color=(255,0,0)):
        # print(canvas.shape)
        for line in self.rlines:
            if line.good:
                canvas = line.draw(canvas, color)
            else:
                canvas = line.draw(canvas, (0,0,128))
        return canvas
    
    def full(self, canvas, color):
        for line in self.rlines:
            if line.good:
                canvas = line.full(canvas, color)
        return canvas

class blockLine:
    def __init__(self):
        self.line = []
        self.ID = 0
        self.len = 0
        self.type = 0
        self.area = 0
        self.direction_line = (0,0)  
        self.center = (0,0)
        self.reshape_center = (0,0)
        self.center_line = (0,0)
        self.isChoose = False
        self.lane_line =0
        self.good = True

    def __init__(self, blockPoint_list):
        self.updata(blockPoint_list)

    def updata(self, blockPoint_list):
        self.line = blockPoint_list

        self.len = len(blockPoint_list)

        self.top = min([y.top for y in blockPoint_list])

        self.down = max([y.down for y in blockPoint_list])

        area = 0
        center = (0,0)
        points = []
        for bp in blockPoint_list:
            bpa = bp.area()
            area = area + bpa
            points.append(bp.Center)
            center = center + bp.Center * bpa
        self.area = area
        self.center = center/area
 
    def check(self, img_in, img_out):
        self.strenth = 0
        uuu = []
        vvv = []
        for bp in self.line:
            self.strenth = self.strenth + bp.check(img_in, img_out)
            if bp.strenth == 2 and not bp.near_edge:
                vvv.append(bp.Center[1])
                uuu.append(bp.Center[0])
        # 检测是否good
        self.good = self.strenth > self.len*2*0.6 and  len(uuu)>2 and len(vvv)>2           
        if self.good:
            z1 = np.polyfit(vvv, uuu, 2)
            self.z1 = z1  # u = v*v * z1[0] + v*z1[1] + z1[2]
            for bp in self.line:
                bp.setCenterZ(z1)

            width = []
            for bp in self.line:    
                if bp.strenth > 0 :
                    width.append((bp.Center[1], bp.width ))
            if len(width) > 1:
                # print(len(width))
                nwidth = np.array(width)
                wid_para = cv2.fitLine(nwidth, cv2.DIST_L2, 0, 1e-2, 1e-2)
                self.width = wid_para[1]/wid_para[0] * (0- wid_para[2]) + wid_para[3]
                self.dwidth = wid_para[1]/wid_para[0]
                # bp.setWidth(self.width, self.dwidth)

                for bp in self.line:    
                    bp.setWidth(self.width, self.dwidth)
            else:
                self.good = False

    def draw(self, canvas, color=(255,0,0)):
        last_bp = None
        if self.good:
            Lcolor = (0,0,255)
        else:
            Lcolor = (0,0,0)
        # Lcolor = (0,0,0)
        for bp in self.line:
            canvas = bp.draw(canvas)
            if last_bp is not None:
                cv2.line(canvas, tuple(bp.Center), tuple(last_bp.Center), Lcolor, 3)
            last_bp = bp
        return canvas

    def draw_raw(self, canvas, color=(255,0,0)):
        last_bp = None

        Lcolor = (0,0,0)
        for bp in self.line:
            canvas = bp.draw(canvas, ccolor=(0,0,0))
            if last_bp is not None:
                cv2.line(canvas, tuple(bp.Center), tuple(last_bp.Center), Lcolor, 2)
            last_bp = bp
        return canvas
    
    def full(self, canvas, color):

        leftPoints = []
        rightPoints = []

        u_top =  self.top * self.top * self.z1[0] + self.top*self.z1[1] + self.z1[2]
        w_top = self.width + self.dwidth*self.top
        u_down = self.down * self.down * self.z1[0] + self.down*self.z1[1] + self.z1[2]
        w_down = self.width + self.dwidth*self.down


        leftPoints.append((u_down - w_down/2, self.down))
        rightPoints.append((u_down + w_down/2, self.down))
        for bp in self.line:
            # canvas = bp.full(canvas,color)
            leftPoints.append((bp.Center[0] - bp.width/2, bp.Center[1]))
            rightPoints.append((bp.Center[0] + bp.width/2, bp.Center[1]))
        leftPoints.append((u_top - w_top/2, self.top))
        rightPoints.append((u_top + w_top/2, self.top))

        points = np.array([leftPoints + rightPoints[: : -1]], dtype = np.int32)
        cv2.fillPoly(canvas, points, color)
        return canvas


class blockPoint:
    def __init__(self, Rank, delta, TContours, imgwith= 560):
        self.indegree_points=[]
        self.outdegree_points=[]
        self.updata(Rank, delta, TContours,imgwith)
        # print(imgwith, self.near_Ledge, self.near_Redge)
    
    def updata(self, Rank, delta, TContours, imgwith = 560):
        self.Rank = Rank
        self.deltaH = delta
        contour = cv2.convexHull(TContours)

        # mu=cv2.moments(contour,False)
        # self.Center=np.array([int(mu['m10'] / (mu['m00']+1e-5)), int(mu['m01'] /(mu['m00']+1e-5) +  Rank * delta)])

        contour = np.squeeze(contour)
        contour = contour + np.array([0, Rank * delta])  # ture coodinate in Image
        y_min = np.min(contour[:,1])
        y_max = np.max(contour[:,1])

        # print(y_min, y_max)
        y_max_thresh = y_max - float(y_max - y_min)/10
        y_min_thresh = y_min + float(y_max - y_min)/10

        down_candi = contour[contour[:,1] > y_max_thresh]
        top_candi = contour[contour[:,1] < y_min_thresh]

        self.up_left_c 	= top_candi[np.argmin(top_candi[:,0])]
        self.up_right_c = top_candi[np.argmax(top_candi[:,0])]
        self.down_left_c = down_candi[np.argmin(down_candi[:,0])]
        self.down_right_c = down_candi[np.argmax(down_candi[:,0])]

        k1 = (y_min - self.up_left_c[1])*1.0/(self.down_left_c[1] - self.up_left_c[1])
        self.up_left = self.up_left_c + (k1 * (self.down_left_c - self.up_left_c)).astype('int64')

        k2 = (y_max - self.up_left_c[1])*1.0/(self.down_left_c[1] - self.up_left_c[1])
        self.down_left = self.up_left_c + (k2 * (self.down_left_c - self.up_left_c)).astype('int64')
        
        k3 = (y_min - self.up_right_c[1])*1.0/(self.down_right_c[1] - self.up_right_c[1])
        self.up_right = self.up_right_c + (k3 * (self.down_right_c - self.up_right_c)).astype('int64')

        k4 = (y_max - self.up_right_c[1])*1.0/(self.down_right_c[1] - self.up_right_c[1])
        self.down_right = self.up_right_c + (k4 * (self.down_right_c - self.up_right_c)).astype('int64')


        # self.up_left 	= top_candi[np.argmin(top_candi[:,0])]       
        # self.down_left = down_candi[np.argmin(down_candi[:,0])]
        # self.down_right = down_candi[np.argmax(down_candi[:,0])]    
        # self.up_right = top_candi[np.argmax(top_candi[:,0])]    



        self.width = (self.down_right[0] + self.up_right[0] 
                        - self.down_left[0] - self.up_left[0])/2
        
        self.height = y_max - y_min
        self.top = y_min
        self.down = y_max

        self.Center = (self.up_left+self.up_right+self.down_left+self.down_right)/4

        self.near_Ledge = self.down_left[0]< 5 and self.up_left[0]<5
        self.near_Redge = self.down_right[0]> imgwith-5 and self.up_right[0]>imgwith-5 
        self.near_edge = self.near_Ledge or self.near_Redge

    def area(self):
        return self.width*self.height

    def corr(self, a, idx):

        I = min(a.down_right[0], self.up_right[0]) - max(a.down_left[0], self.up_left[0])

        if self.Rank - a.Rank <= 0 or self.Rank - a.Rank > 5:
            cor = 0
        elif self.Rank - a.Rank == 1  and I< self.width * 0.1:
            cor = 0
        else:
            cor = np.linalg.norm(a.Center - self.Center)
            self.outdegree_points.append((cor,idx))
        
        return cor       

    def closet_Point():

        pass

    def sort(self):
        self.pre_ID_seq.sort()
        self.last_ID_seq.sort()

    def check(self, img_in, img_out):
        # img_in --- rail in block
        # img_out -- ballast surrand block

        points_left = [self.down_left, 
                        ((self.down_right[0]+self.down_left[0]*2)/3,self.down_right[1]),
                        ((self.up_right[0]+self.up_left[0]*2)/3,self.up_right[1]),
                        self.up_left]
        points_right = [self.down_right, 
                        ((self.down_right[0]*2+self.down_left[0])/3,self.down_right[1]),
                        ((self.up_right[0]*2+self.up_left[0])/3,self.up_right[1]),
                        self.up_right]
        # mask_all = np.zeros_like(img_in)               
        mask_right = np.zeros_like(img_in)
        mask_left = np.zeros_like(img_in)
        # print('left',points_left)
        # print('right',points_right)
        # cv2.fillConvexPoly(mask_all, np.array([points_all], dtype = 'int'), ( 255))
        cv2.fillConvexPoly(mask_left, np.array([points_left], dtype = 'int'), ( 255))
        cv2.fillConvexPoly(mask_right, np.array([points_right], dtype = 'int'), ( 255))

        # rst_in_left = cv2.bitwise_and(img_in, img_in, mask=mask_all)


        rst_in_left = cv2.bitwise_and(img_in, img_in, mask=mask_left)
        rst_in_right = cv2.bitwise_and(img_in, img_in, mask=mask_right)

        # cv2.imshow('left',rst_in_left)
        # cv2.imshow('right',rst_in_right)
        # cv2.waitKey(0)
        # sum_left = np.sum(np.sum(rst_in_left))/255
        # sum_right = np.sum(np.sum(rst_in_right))/255

        self.strenth = self.check_rail(rst_in_left) + self.check_rail(rst_in_right)
        return self.strenth

    def check_rail(self, rail_mask):
        sum = np.sum(np.sum(rail_mask))/255
        # ww = np.sum(np.max(rail_mask, axis=0))/255
        ww = np.max(np.sum(rail_mask, axis=1))/255
        hh = np.sum(np.max(rail_mask, axis=1))/255


        if ww > self.width * 0.05 and ww < self.width*0.3 and hh > self.height * 0.8:            
            if sum > ww*hh*0.5:
                # print('good: ',sum, ww, hh)
                return 1
            else:
                # print('no rail track',self.width, self.height)
                # print(sum, ww, hh)   
                pass 

        return 0

    def setWidth(self, width0, dwidth):
        self.width = width0 + dwidth*self.Center[1]

    def setCenterZ(self, z1):
        yyy = self.Center[1]
        xxx = yyy * yyy * z1[0] + yyy*z1[1] + z1[2]
        self.Center = np.array((xxx, yyy), dtype = 'int32')
        if self.strenth >0:
            if self.near_Ledge:
                self.width = 2*((self.down_right[0] + self.up_right[0])/2 - xxx)
            if self.near_Redge:
                self.width = 2*(xxx - (self.down_left[0] + self.up_left[0])/2 )
        # print(self.width)

    def draw(self, canvas, lcolor=(0,0,0), ccolor=(0,0,255), thinkness = 2):
        if self.strenth == 0:
            lcolor=(0.5,0.5,0.5)
        elif self.strenth == 1:
            lcolor=(0.5,0.5,0.5)
        
        if self.near_Ledge:
            Lcolor =(128,0,128)
        else:
            Lcolor = lcolor
        if self.near_Redge:
            Rcolor =(128,0,128)
        else:
            Rcolor = lcolor
        # print(canvas.shape, canvas.dtype)
        cv2.line(canvas,    tuple(self.down_left),tuple(self.down_right), lcolor,thinkness)
        cv2.line(canvas,    tuple(self.down_right),tuple (self.up_right), Rcolor,thinkness)
        cv2.line(canvas,    tuple(self.up_right), tuple(self.up_left), lcolor,thinkness)
        cv2.line(canvas,    tuple(self.up_left), tuple(self.down_left), Lcolor,thinkness)
        cv2.circle(canvas,  tuple(self.Center), 5, ccolor,-1)

        return canvas


# def my_compare(A, B):
#     if A[2] == B[2]:
#         return int(1000*A[1]/(A[3]+1e-8) - 1000*B[1]/(B[3]+1e-8))
#     else:
#         return B[2] - A[2] 

def read_img(filename, iscolor= 1):
    img = cv2.imread(filename, iscolor )
    # cv2.imshow('fillMap', img)
    if iscolor:
        B= img[:,:,0]
        G= img[:,:,1]
        R= img[:,:,2]
        B[B<50] = 0
        B[B>=50] = 1
        G[G<50] = 0
        G[G>=50] = 2
        R[R<20] = 0
        R[R>=20] = 3
        img = B + G + R
    # cv2.imshow('b', B)
    # cv2.imshow('g', G)
    # cv2.imshow('r', R)
    img = img.astype('uint8')
    img1 = np.zeros_like(img)
    img1[img>=1] = 255

    # img1=cv2.erode(img1, (3,3))
    # img1=cv2.dilate(img1, (3,3))

    img2 = np.zeros_like(img)
    img2[img>=2] = 255
    

    retval=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img3 = np.zeros_like(img)
    img3[img>=3] = 255

    img3=cv2.erode(img3,retval)

    # img3=cv2.dilate(img3, retval)


    return [img1, img2, img3]

# image_dir = r"..\result_summary\model_at\CSPs-AL-at"
image_dir = r"."
def main():
    segmenter = railSegmenter()
    file_list = os.listdir(image_dir)
    for file in file_list:
        if file.endswith('_color.jpg'):
            imagePth = os.path.join(image_dir,file)
            img = read_img(imagePth)
            mask_batch, draw_batch = segmenter.run(np.array([img]), 255)
            for idx, (maks, draw) in enumerate( zip(mask_batch,draw_batch)):
                drawName = os.path.join(image_dir,file[:-10]+'_draw.jpg')
                maskName = os.path.join(image_dir,file[:-10]+'_railmask.jpg')
                cv2.imshow('draw', draw)
                cv2.imshow('mask', maks)
                cv2.waitKey()
                cv2.imwrite( drawName,draw)
                cv2.imwrite( maskName,maks)

def main_one_img(filename):
    segmenter = railSegmenter()
 
    if filename.endswith('_color.jpg'):
        imagePth0 = filename[:-10]+'.jpg'
        print(filename)
        print(imagePth0)
        if not os.path.isfile(imagePth0):
            return
        # print(imagePth0)
        lablemask = read_img(filename)
        canvas = cv2.imread(imagePth0)
        rawImage = canvas.copy()#cv2.resize(canvas, (576, 416))
        # canvas = canvas.transpose((2,0,1))
        maskLabel_batch,blockLines_batch,blockPoints_batch,maskImage_batch = segmenter.run(np.array([rawImage]),np.array([lablemask]), 255)#,np.array([canvas]))
        for idx, (mkCls, bLine, bPnt, mkImg) in enumerate( zip(maskLabel_batch,blockLines_batch,blockPoints_batch,maskImage_batch)):
            cv2.imshow('draw', bLine)
            cv2.imshow('mask', mkCls)
            cv2.imshow('bolck', bPnt)
            cv2.imshow('bd', mkImg/255)
            cv2.waitKey()


def main_one():
    segmenter = railSegmenter()
    file_list = os.listdir(image_dir)
    for file in file_list:
        if file.endswith('_color.jpg'):
            imagePth0 = os.path.join(image_dir,file[:-10]+'_org.png')
            if not os.path.isfile(imagePth0):
                continue
            print(imagePth0)
            imagePth = os.path.join(image_dir,file)
            lablemask = read_img(imagePth)
            canvas = cv2.imread(imagePth0)
            rawImage = cv2.resize(canvas, (720, 576))
            # canvas = canvas.transpose((2,0,1))
            mask_batch, draw_batch, block_batch, bd_batch = segmenter.run(np.array([rawImage]),np.array([lablemask]), 255)#,np.array([canvas]))
            for idx, (maks, draw, block, bd) in enumerate( zip(mask_batch,draw_batch,block_batch,bd_batch)):
                cv2.imshow('draw', draw)
                cv2.imshow('mask', maks)
                cv2.imshow('bolck', block)
                cv2.imshow('bd', bd)
                cv2.waitKey()
                drawName = os.path.join(image_dir,file[:-10]+'_draw0.jpg')
                maskName = os.path.join(image_dir,file[:-10]+'_railmask.jpg')
                bolckName = os.path.join(image_dir,file[:-10]+'_block.jpg')
                bdkName = os.path.join(image_dir,file[:-10]+'_bd.jpg')
                cv2.imwrite( drawName,draw)
                cv2.imwrite( maskName,maks)
                cv2.imwrite( bolckName,block)
                cv2.imwrite( bdkName,bd)

def showTask():
    label = cv2.imread('figure/fig_6b.png', 0)
    # print((label.shape[0],label.shape[1],3))
    BGR = np.zeros((label.shape[0],label.shape[1],3))+128
    B = BGR[:,:,0]
    G = BGR[:,:,1]
    R = BGR[:,:,2]
    BGR[label == 1] = [255,0,0] 
    BGR[label == 2] = [0,255,0] 
    BGR[label >= 3] = [0,0,255] 
    filename = 'lable_color.jpg'
    cv2.imwrite( filename,BGR)

    BGR[label <= 1 ] = [128,128,128] 
    BGR[label > 1] = [255,255,255] 
    filename = 'rail_color.jpg'
    cv2.imwrite( filename,BGR)


if __name__ == "__main__":
    # main()
    # main_one()
    # showTask()
    # main_one_img(r'run/iRailway/CSPnet-A-S-ce-iRailway/output/000010_color.jpg')
    main_one_img(r'samples/000026_color.jpg')




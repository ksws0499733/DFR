# -*- coding: utf-8 -*
import cv2.cv2 as cv2
import numpy as np
from functools import cmp_to_key
import os

# global gnumber = 0

class railSegmenter:
    def __init__(self):
        self.rlines = []
        self.gnumber = 0

    def run(self,img_batch, color = 1, canvas_batch=None):
        # print('0',img_batch.shape)
        if len(img_batch.shape) == 3:
            mask_batch = np.zeros((img_batch.shape[0],img_batch.shape[1],img_batch.shape[2]))
            if canvas_batch is None:
                print('0',img_batch.shape)
                canvas_batch = np.zeros((img_batch.shape[0],img_batch.shape[1],img_batch.shape[2],3), dtype = 'uint8')+220

        elif len(img_batch.shape) == 4:
            mask_batch = np.zeros((img_batch.shape[0],img_batch.shape[2],img_batch.shape[3]))
            if canvas_batch is None:
                print('1',img_batch.shape)
                canvas_batch = np.zeros((img_batch.shape[0],img_batch.shape[2],img_batch.shape[3],3), dtype = 'uint8')+220
        else:
            print('input data has no enough dims')
            return None, None


        for id, img in enumerate(img_batch):
            if len(img.shape) == 2:
                img0 = np.zeros_like(img)
                img0[img>=1] = 255

                img1 = np.zeros_like(img)
                img1[img>=2] = 255
            
                img2 = np.zeros_like(img)
                img2[img>=3] = 255
            else:
                img0 =img[0]
                img1 =img[1]
                img2 =img[2]



            imgs = np.split(img1, 16, axis= 0)
            bPoint_list = []
            for idx0, sub_img in enumerate(imgs):
                
                h,w = sub_img.shape
                # print(w,h)
                contours, _ = cv2.findContours(sub_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                # print(idx0, ' = ',len(contours))
                for contour in contours:
                    if cv2.contourArea(contour) > 200:
                        bLine = blockPoint(idx0, h, contour)
                        bPoint_list.append(bLine)

            self.updata(bPoint_list)
            self.check(img2,img0)
            self.full(mask_batch[id,:,:], color)
            canvas = canvas_batch[id]
            self.draw(canvas)
        return mask_batch,canvas_batch

    def run2(self,img_batch, label_batch, color = 1, canvas_batch=None, is_draw = True):
        
        # print('img_batch',img_batch.shape)
        # print('label_batch',label_batch.shape)
        
        
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
            # print('0',label_batch.shape)
            canvas_batch = np.zeros((N,H,W,3), dtype = 'uint8')+220

        for id, (img, lable) in enumerate(zip(img_batch, label_batch)):
            # print(len(lable.shape))
            # print('current id:',self.gnumber)
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


            temp = np.zeros((H,W,3), dtype = 'uint8')+220
            for bp in rPoint_list:
                bp.check(lable2,lable0)
                bp.draw(temp, color=(0,0,255))
            block_batch[id] = temp

            # temp2 = np.zeros((H,W,3), dtype = 'uint8')+220
            # for bp in tPoint_list:
            #     bp.draw(temp2, color=(0,0,255))
            # # cv2.imshow('2',temp2)
            # # cv2.waitKey(0)            
            
            self.blines = self.find_blockLines(bPoint_list)
            self.rlines = self.find_blockLines(rPoint_list)
            self.tlines = self.find_blockLines(tPoint_list)

            # for track in self.tlines:
            #     track.fitLines()
            #     canvas = track.draw(temp)
            #     # cv2.imshow('5',canvas)
            #     # cv2.waitKey(0) 

            for line in self.rlines:
                line.check(lable2, lable0)

            # self.check(lable2,lable0)
            self.full(mask_batch[id,:,:], color)
            
            for line in self.rlines:
                if line.good:
                    bdmap_batch[id] = line.drawbound(img, color)
            # cv2.imshow('3',bdmap)
            # cv2.waitKey(0)   
            # print("canvas ", canvas.shape)
            self.draw(canvas_batch[id])
        return mask_batch,canvas_batch,block_batch,bdmap_batch

    def find_blockPoints(self, label, bin_number=16, axis=0):
        W = label.shape[1]
        imgs = np.split(label, bin_number, axis= axis)
        bPoint_list = []
        for idx0, sub_img in enumerate(imgs):
            
            h,w = sub_img.shape
            # print(w,h)
            contours, _ = cv2.findContours(sub_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # print(idx0, ' = ',len(contours))
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    bLine = blockPoint(idx0, h, contour, W)
                    bPoint_list.append(bLine)
        return bPoint_list

    def run4train(self,label_batch, color = 1):
        if len(label_batch.shape) == 3:
            W = label_batch.shape[2]
            mask_batch = np.zeros((label_batch.shape[0],label_batch.shape[1],label_batch.shape[2]))


        elif len(label_batch.shape) == 4:
            W = label_batch.shape[3]
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

            # cv2.imshow('r',img)
            # cv2.imshow('0',lable0)
            # cv2.imshow('1',lable1)
            # cv2.imshow('2',lable2)
            # cv2.waitKey(0)


            imgs = np.split(lable1, 16, axis= 0)
            bPoint_list = []
            for idx0, sub_img in enumerate(imgs):
                
                h,w = sub_img.shape
                # print(w,h)
                contours, _ = cv2.findContours(sub_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                # print(idx0, ' = ',len(contours))
                for contour in contours:
                    if cv2.contourArea(contour) > 200:
                        bLine = blockPoint(idx0, h, contour, W)
                        bPoint_list.append(bLine)

            # for bp in bPoint_list:
            #     bp.check(lable2,lable0) 
            # print("bPoint_list",len(bPoint_list))        
            
            self.updata2(bPoint_list)
            self.check(lable2,lable0)
            self.full(mask_batch[id,:,:], color)
            
        return mask_batch

    def check(self, img_in, img_out):
        for line in self.lines:
            line.check(img_in, img_out)
        pass


    def updata(self, blockPoint_list):
        self.lines = []
        corMat = 0
        for id1, bp1 in enumerate(blockPoint_list) :
            for id2, bp2 in enumerate(blockPoint_list):
                cor = bp1.corr(bp2, id2)
                # print(id1, id2, cor)
            bp1.sort()

        while 1:
            line_info = []
            for id1, bp1 in enumerate(blockPoint_list) :

                if len(bp1.pre_ID_seq) == 0 and len(bp1.last_ID_seq)>0 and bp1.isUsed == False:
                    total_cor = 0
                    total_len = 0
                    actul_len = 0
                    newPoint = bp1
                    nextPoint = newPoint
                    newList = []
                    newList.append(id1)
                    while len(newPoint.last_ID_seq)>0:
                        is_find_next = False
                        for cor,idx in newPoint.last_ID_seq:
                            if blockPoint_list[idx].isUsed == False:
                                nextPoint = blockPoint_list[idx]
                                newList.append(idx)
                                total_cor = total_cor + cor
                                total_len = total_len + abs(newPoint.Rank - nextPoint.Rank)
                                actul_len = actul_len + 1
                                newPoint = nextPoint
                                is_find_next = True
                                break
                        if is_find_next == False:
                            break
                    if(actul_len > 0):
                        line_info.append((newList,total_cor,total_len, actul_len ))

            if len(line_info) == 0:
                break
            else:
                line_info.sort(key= cmp_to_key(my_compare))  # find longest and minimum corr

                if line_info[0][2] < 6:
                    break
                else:
                    newList = []
                    for idx in line_info[0][0]:
                        newList.append(blockPoint_list[idx])
                        blockPoint_list[idx].isUsed = True
                    self.lines.append(blockLine(newList))
    
    def updata2(self, blockPoint_list):
        
        NN = len(blockPoint_list)
        corMat = np.zeros((NN,NN))
        for id1, bp1 in enumerate(blockPoint_list) :
            for id2, bp2 in enumerate(blockPoint_list):
                cor = bp1.corr(bp2, id2)
                if bp1.Rank - bp2.Rank < 0 :
                    corMat[id1, id2] = cor
                else:
                    corMat[id1, id2] = 1e8
                # print(id1, id2, cor)
            bp1.sort()
        

        head_list = []
        tail_list = []
        line_list = []
        for id1, bp1 in enumerate(blockPoint_list) :
            if len(bp1.pre_ID_seq) == 0:
                head_list.append(id1)
                line_list.append([id1])
            if len(bp1.last_ID_seq) == 0:
                tail_list.append(id1)
        # print('head_list',head_list)
        # print('tail_list',tail_list)
        while 1:
            link_list = []
            link_list_cor = []
            for ii, idh in enumerate(head_list):
                list_a = corMat[idh].tolist()
                min_list = min(list_a) #返回最大值
                min_index = list_a.index(min(list_a)) # 返回最大值的索引
                if min_list < 1e7:
                    link_list.append((ii,idh,min_index, min_list))
                    link_list_cor.append(min_list)
            # print('link_list',link_list) 
            if len(link_list) == 0:
                break
            min_ii = link_list_cor.index(min(link_list_cor)) # 返回最大值的索引
            line_list[link_list[min_ii][0]].append(link_list[min_ii][2])
            head_list[link_list[min_ii][0]] = link_list[min_ii][2]
            corMat[:,link_list[min_ii][2]] = 1e8

            # os.system("pause")
        lines = []
        for line in line_list:
            newList = []
            for idx in line:
                newList.append(blockPoint_list[idx])
            lines.append(blockLine(newList))
        return lines
    
    def find_blockLines(self, blockPoint_list):
        
        NN = len(blockPoint_list)
        corMat = np.zeros((NN,NN))
        for id1, bp1 in enumerate(blockPoint_list) :
            for id2, bp2 in enumerate(blockPoint_list):
                cor = bp1.corr(bp2, id2)
                if bp1.Rank - bp2.Rank < 0 :
                    corMat[id1, id2] = cor
                else:
                    corMat[id1, id2] = 1e8
                # print(id1, id2, cor)
            bp1.sort()
        

        head_list = []
        tail_list = []
        line_list = []
        for id1, bp1 in enumerate(blockPoint_list) :
            if len(bp1.pre_ID_seq) == 0:
                head_list.append(id1)
                line_list.append([id1])
            if len(bp1.last_ID_seq) == 0:
                tail_list.append(id1)
        # print('head_list',head_list)
        # print('tail_list',tail_list)
        while 1:
            link_list = []
            link_list_cor = []
            for ii, idh in enumerate(head_list):
                list_a = corMat[idh].tolist()
                min_list = min(list_a) #返回最大值
                min_index = list_a.index(min(list_a)) # 返回最大值的索引
                if min_list < 1e7:
                    link_list.append((ii,idh,min_index, min_list))
                    link_list_cor.append(min_list)
            # print('link_list',link_list) 
            if len(link_list) == 0:
                break
            min_ii = link_list_cor.index(min(link_list_cor)) # 返回最大值的索引
            line_list[link_list[min_ii][0]].append(link_list[min_ii][2])
            head_list[link_list[min_ii][0]] = link_list[min_ii][2]
            corMat[:,link_list[min_ii][2]] = 1e8

            # os.system("pause")
        lines = []
        for line in line_list:
            newList = []
            for idx in line:
                newList.append(blockPoint_list[idx])
            lines.append(blockLine(newList))
        return lines
    


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
        self.strenth = 0
        self.bound = []
        self.updata(blockPoint_list)
        self.width = 0 #width of head block
        self.dwidth = 0 #width diffrence of near block
        self.center0 = 0 #width of head block
        self.dcenter = 0 #width diffrence of near block
    
    def updata(self, blockPoint_list):
        self.line = blockPoint_list
        self.len = len(blockPoint_list)
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
        points = np.array(points)
        line_para = cv2.fitLine(points, cv2.DIST_L2, 0, 1e-2, 1e-2)
        direction_line = (line_para[0]*100,  line_para[1]*100)
        if(direction_line[1] > 0):
            direction_line = (-direction_line[0], -direction_line[1])
        self.direction_line = direction_line
    
    def check(self, img_in, img_out):
        self.strenth = 0
        xxx = []
        yyy = []
        for bp in self.line:
            self.strenth = self.strenth + bp.check(img_in, img_out)
            if bp.strenth == 2 and not bp.near_edge:
                xxx.append(bp.Rank)
                yyy.append(bp.Center[0])
        # 检测是否good
        self.good = self.strenth > self.len*2*0.6 and  len(xxx)>2 and len(yyy)>2           

        if self.good:
            z1 = np.polyfit(xxx, yyy, 2)
            self.z1 = z1
            for bp in self.line:
                bp.setCenterZ(z1)

            width = []
            for bp in self.line:    
                if bp.strenth > 0 :
                    width.append([bp.Rank, bp.width ])
            if len(width) > 1:
                # print(len(width))
                nwidth = np.array(width, dtype=np.float32)
                try:
                    wid_para = cv2.fitLine(nwidth, cv2.DIST_L2, 0, 1e-2, 1e-2)
                except:
                    print('fitLine error: ',nwidth)
                    wid_para = [0,1,0,0]

                self.width = wid_para[1]/wid_para[0] * (0- wid_para[2]) + wid_para[3]
                self.dwidth = wid_para[1]/wid_para[0]
                bp.setWidth(self.width, self.dwidth)

                for bp in self.line:    
                    bp.setWidth(self.width, self.dwidth)
            else:
                self.good = False

    def fitLines(self):
        xxx = []
        yyy = []
        width = []
        for bp in self.line:
            xxx.append(bp.Rank)
            yyy.append(bp.Center[0])
            width.append((bp.Rank, bp.width ))
        # 检测是否good
        print('===1===:',xxx,yyy)
        z1 = np.polyfit(xxx, yyy, 2)
        self.z1 = z1
        for bp in self.line:
            bp.setCenterZ(z1)
        try:
            nwidth = np.array(width.copy())
            wid_para = cv2.fitLine(nwidth, cv2.DIST_L2, 0, 1e-2, 1e-2)
            self.width = wid_para[1]/wid_para[0] * (0- wid_para[2]) + wid_para[3]
            self.dwidth = wid_para[1]/wid_para[0]
        except Exception:
            print("Exception:",Exception,"nwidth:",nwidth)

        for bp in self.line:
            bp.setWidth(self.width, self.dwidth)


    def draw(self, canvas, color=(255,0,0)):
        last_bp = None
        if self.good:
            Lcolor = (0,0,255)
        else:
            Lcolor = (0,0,0)
        for bp in self.line:
            canvas = bp.draw(canvas)
            if last_bp is not None:
                cv2.line(canvas, tuple(bp.Center), tuple(last_bp.Center), Lcolor, 3)
            last_bp = bp
        return canvas
    
    def full(self, canvas, color):

        leftPoints = []
        rightPoints = []

        center_top = self.center0 + self.dcenter*(self.line[0].Rank-0.5)
        center_down = self.center0 + self.dcenter*(self.line[-1].Rank+0.5)
        xxx = self.line[0].Rank-0.5
        center_top = xxx * xxx * self.z1[0] + xxx*self.z1[1] + self.z1[2]
        xxx = self.line[-1].Rank+0.5
        center_down = xxx * xxx * self.z1[0] + xxx*self.z1[1] + self.z1[2]

        width_top = self.width + self.dwidth*(self.line[0].Rank-0.5)
        width_down = self.width + self.dwidth*(self.line[-1].Rank+0.5)

        leftPoints.append((center_top - width_top/2, self.line[0].up_left[1]))
        rightPoints.append((center_top + width_top/2, self.line[0].up_left[1]))

        # leftPoints.append(self.line[0].up_left)
        # rightPoints.append(self.line[0].up_right)

        for bp in self.line:
            # canvas = bp.full(canvas,color)
            leftPoints.append((bp.Center[0] - bp.width/2, bp.Center[1]))
            rightPoints.append((bp.Center[0] + bp.width/2, bp.Center[1]))
            
        
        leftPoints.append((center_down - width_down/2, self.line[-1].down_left[1]))
        rightPoints.append((center_down + width_down/2, self.line[-1].down_left[1]))

        points = np.array([leftPoints + rightPoints[: : -1]], dtype = np.int32)
        cv2.fillPoly(canvas, points, color)
        return canvas

    def drawbound(self, canvas, color):

        leftPoints = []
        rightPoints = []

        # print(canvas.shape)

        bdMap = np.zeros((canvas.shape[0],canvas.shape[1],3))
        mask = np.zeros((canvas.shape[0],canvas.shape[1],3))

        center_top = self.center0 + self.dcenter*(self.line[0].Rank-0.5)
        center_down = self.center0 + self.dcenter*(self.line[-1].Rank+0.5)
        xxx = self.line[0].Rank-0.5
        center_top = xxx * xxx * self.z1[0] + xxx*self.z1[1] + self.z1[2]
        xxx = self.line[-1].Rank+0.5
        center_down = xxx * xxx * self.z1[0] + xxx*self.z1[1] + self.z1[2]

        width_top = self.width + self.dwidth*(self.line[0].Rank-0.5)
        width_down = self.width + self.dwidth*(self.line[-1].Rank+0.5)

        leftPoints.append((center_top - width_top/2, self.line[0].up_left[1]))
        rightPoints.append((center_top + width_top/2, self.line[0].up_left[1]))

        # leftPoints.append(self.line[0].up_left)
        # rightPoints.append(self.line[0].up_right)

        for bp in self.line:
            # canvas = bp.full(canvas,color)
            leftPoints.append((bp.Center[0] - bp.width/2, bp.Center[1]))
            rightPoints.append((bp.Center[0] + bp.width/2, bp.Center[1]))
            
        
        leftPoints.append((center_down - width_down/2, self.line[-1].down_left[1]))
        rightPoints.append((center_down + width_down/2, self.line[-1].down_left[1]))

        points = np.array([leftPoints + rightPoints[: : -1]], dtype = np.int32)
        cv2.fillPoly(mask, points, (255,255,255))
        # cv2.imshow('4',canvas)
        # cv2.waitKey(0)   
        # print(canvas.shape)
        # print(mask.shape)

        bdMap = canvas/2 + mask/2
        # cv2.imshow('5',mask)
        # cv2.waitKey(0)  
        # bdMap = bdMap/255
        # cv2.imshow('5',bdMap)
        # cv2.waitKey(0)          
        # = cv2.addWeighted(canvas,0.5,mask,0.5,0)

        cv2.polylines(bdMap,np.array([leftPoints], dtype= np.int32), False, (255,255,0), thickness=3 )
        cv2.polylines(bdMap,np.array([rightPoints], dtype= np.int32), False, (0x33,0xa3,0xdc), thickness=3 )
        # cv2.imshow('bd',bdMap)
        # cv2.waitKey(0)

        return bdMap

class blockPoint:

    def __init__(self, Rank, delta, TContours, imgwith= 560):
        self.Rank = 0
        self.delta = 0
        self.Center = np.array([0,0])
        self.Center_up = (0,0)
        self.Center_down = (0,0)
        self.endPoint = np.array([[0,0],[0,0],[0,0],[0,0]])
        self.down_left = (0,0)
        self.down_right = (0,0)
        self.up_left 	= (0,0)
        self.up_right = (0,0)

        self.width = 0
        self.height = 0

        self.isUsed = False

        self.pre_ID_seq = []
        self.last_ID_seq = []

        self.strenth = 0 # 0: weak, 1 middle, 2 strong
        self.near_Ledge = 0 # 
        self.near_Redge = 0 # 
        self.near_edge = 0 # 
        
        self.updata(Rank, delta, TContours,imgwith)
        # print(imgwith, self.near_Ledge, self.near_Redge)
    
    def updata(self, Rank, delta, TContours, imgwith = 560):
        self.Rank = Rank
        self.delta = delta
        contour = cv2.convexHull(TContours)

        mu=cv2.moments(contour,False)
        self.Center=np.array([int(mu['m10'] / (mu['m00']+1e-5)), int(mu['m01'] /(mu['m00']+1e-5) +  Rank * delta)])

        contour = np.squeeze(contour)
        contour = contour + np.array([0, Rank * delta])
        y_min = np.min(contour[:,1])
        y_max = np.max(contour[:,1])

        # print(y_min, y_max)
        y_max_thresh = y_max - float(y_max - y_min)/10
        y_min_thresh = y_min + float(y_max - y_min)/10

        down_left_thresh = np.min(contour[contour[:,1] > y_max_thresh,0])
        down_right_thresh = np.max(contour[contour[:,1] > y_max_thresh,0])
        up_left_thresh 	= np.min(contour[contour[:,1] < y_min_thresh,0])
        up_right_thresh = np.max(contour[contour[:,1] < y_min_thresh,0])  

        self.down_left = (down_left_thresh,Rank * delta +delta)
        self.down_right = (down_right_thresh,Rank * delta +delta)
        self.up_left 	= (up_left_thresh,Rank * delta)
        self.up_right = (up_right_thresh,Rank * delta)

        self.width = (down_right_thresh + up_right_thresh - down_left_thresh - up_left_thresh)/2
        self.height =delta

        self.near_Ledge = down_left_thresh< 5 and up_left_thresh<5
        self.near_Redge = down_right_thresh> imgwith-5 and up_right_thresh>imgwith-5 
        self.near_edge = self.near_Ledge or self.near_Redge

        # self.endPoint = np.array([down_left,down_right,up_left,up_right])
    
    def area(self):
        return (self.up_right[0] - self.up_left[0] +self.down_right[0] - self.down_left[0])*self.delta/2

    def corr(self, a, idx):
        cor = 1e8
        if (self.Rank - a.Rank > 0 and self.Rank - a.Rank < 5):
            up_right_r = self.up_right[0]# - (Rank - a.Rank - 1.0)* direct
            up_left_r = self.up_left[0]# - (Rank - a.Rank - 1.0) * direct
            I = min(a.down_right[0], up_right_r) - max(a.down_left[0], up_left_r)
            if I < self.width * 0.5:
                return 1e8
            R = abs(a.down_left[0] - up_left_r) + abs(a.down_right[0] - up_right_r)
            cor = np.linalg.norm(a.Center - self.Center)# * R / I +self.Rank - a.Rank
            self.pre_ID_seq.append([idx, cor])
        elif (self.Rank - a.Rank < 0 and self.Rank - a.Rank > -5):
            up_right_r = a.up_right[0]# + (a.Rank - Rank - 1) * a.direct
            up_left_r = a.up_left[0]# + (a.Rank - Rank - 1) * a.direct
            # print(up_right_r, up_left_r)
            # print(min(self.down_right[0], up_right_r))
            # print(max(self.down_left[0], up_left_r))
            I = min(self.down_right[0], up_right_r) - max(self.down_left[0], up_left_r)
            if I < self.width * 0.5:
                return 1e8
            R = abs(self.down_left[0] - up_left_r) + abs(self.down_right[0] - up_right_r)
            cor = np.linalg.norm(a.Center - self.Center)# * R / (I+1e-5) - self.Rank + a.Rank
            self.last_ID_seq.append([cor, idx])     
        return cor       

    def sort(self):
        self.pre_ID_seq.sort()
        self.last_ID_seq.sort()

    def check(self, img_in, img_out):
        # in——rail in block
        # out--ballast surrand block

        # mask = np.zeros_like(img_in)
        # points = [self.down_left, self.down_right,self.up_right,self.up_left]
        # cv2.fillPoly(mask, [points], ( 255), 8, 0)
        # rst_out = cv2.bitwise_and(img_out, img_out, mask=mask)
        # rst_in = cv2.bitwise_and(img_in, img_in, mask=mask)
        # points_all = [self.down_left, 
        #                 self.down_right[1],
        #                 self.up_right[1],
        #                 self.up_left]   
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
        self.width = width0 + dwidth*self.Rank

    def setCenter(self, center0, dcenter):
        center = center0 + dcenter*self.Rank
        self.Center = np.array((center, self.Center[1]), dtype = 'int32')

    def setCenterZ(self, z1):
        xxx = self.Rank
        center = xxx * xxx * z1[0] + xxx*z1[1] + z1[2]
        self.Center = np.array((center, self.Center[1]), dtype = 'int32')
        if self.strenth >0:
            if self.near_Ledge:
                self.width = 2*((self.down_right[0] + self.up_right[0])/2 - center)
            if self.near_Redge:
                self.width = 2*(center - (self.down_left[0] + self.up_left[0])/2 )
        # print(self.width)

    def draw(self, canvas, color=(0,0,0), thinkness = 2):
        if self.strenth == 0:
            color=(0.5,0.5,0.5)
        elif self.strenth == 1:
            color=(0.5,0.5,0.5)
        
        if self.near_Ledge:
            Lcolor =(128,0,128)
        else:
            Lcolor = color
        if self.near_Redge:
            Rcolor =(128,0,128)
        else:
            Rcolor = color
        # print(canvas.shape, canvas.dtype)
        cv2.line(canvas, self.down_left, self.down_right, color,thinkness)
        cv2.line(canvas, self.down_right, self.up_right, Rcolor,thinkness)
        cv2.line(canvas, self.up_right, self.up_left, color,thinkness)
        cv2.line(canvas, self.up_left, self.down_left, Lcolor,thinkness)
        cv2.circle(canvas, tuple(self.Center), 5, (0,0,255),-1)

        # if self.near_Ledge:
        #     Lcolor =(128,0,128)
        # else:
        #     Lcolor = color
        # if self.near_Redge:
        #     Rcolor =(128,0,128)
        # else:
        #     Rcolor = color
        # cv2.line(canvas, self.down_left, self.down_right, color,thinkness)
        # cv2.line(canvas, self.down_right, self.up_right, Rcolor,thinkness)
        # cv2.line(canvas, self.up_right, self.up_left, color,thinkness)
        # cv2.line(canvas, self.up_left, self.down_left, Lcolor,thinkness)
        # cv2.circle(canvas, tuple(self.Center), 5, (0,0,255),-1)

        return canvas

    def full(self, canvas, color):
        points = np.array([[self.down_left, self.down_right,self.up_right,self.up_left]], dtype = np.int32)
        # points = np.array([[[100,100], [200,230], [150,200], [100,220]]], dtype = np.int32)

        # print(pointsb)
        # print(points)

        cv2.fillConvexPoly(canvas, points, color)
        return canvas


def my_compare(A, B):
    if A[2] == B[2]:
        return int(1000*A[1]/(A[3]+1e-8) - 1000*B[1]/(B[3]+1e-8))
    else:
        return B[2] - A[2] 

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


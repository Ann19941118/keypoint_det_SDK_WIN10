from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from genericpath import exists

from PIL.Image import Image
import numpy as np

import os
import cv2

from scipy import optimize
from skimage import morphology
import numpy as np

from datasets.dataset_factory import dataset_factory
from opts import opts
from detectors.detector_factory import detector_factory
import warnings
warnings.filterwarnings('ignore')

import yaml

edges = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], 
             [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49], 
             [50, 51], [52, 53], [54, 55], [56, 57], [58, 59], [60, 61], [62, 63], [64, 63]]

edges_new = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], 
             [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49], 
             [50, 51], [52, 53], [54, 55], [56, 57], [58, 59], [60, 61], [62, 63], [64, 65]]

def f_1(x, A, B):
    return A * x + B


class Stage2_det(object):
    def __init__(self):
        self.opt = opts().init()
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str
        self.detector = self.model_load()
        self.image_list = []
    
    def model_load(self):
        Dataset = dataset_factory[self.opt.dataset]
        self.opt = opts().update_dataset_info_and_set_heads(self.opt, Dataset)

        Detector = detector_factory[self.opt.task]
        detector = Detector(self.opt)
        return detector

    def load_config(self,config_path):
        try:
            with open(config_path) as f:
                conf = yaml.load(f)

            self.conf = conf['configuration']['algorithm']
            self.thresh = conf['configuration']['detect']['Align']['thresh']
        except:
            self.conf = {'AnodeDL':{'threshold': 1},
                          'CathodeDL':{'threshold': 1},
                          'DL': {'threshold': 4.5},
                          'low_limit': 0.52,
                          'pixel':{'1': 0.022,'2': 0.02,'3': 0.016,'4': 0.014},
                          'test_points':{'1': 'Test_Point_A','2': 'Test_Point_B','3': 'Test_Point_C','4': 'Test_Point_D'},
                          'up_limit':{'1': 2.62,'2': 2.97,'3': 2.97,'4': 2.62}}
            self.thresh = 0.2

    def detect(self,img_dict,rank):
        new_dict = {}
        for img_name,v in img_dict.items():

            img_crop = v[0]
            xmin_,ymin_ = v[1],v[2]
            img_rotate = v[3]
            dst = cv2.Sobel(img_crop[:,:,0], cv2.CV_64F,1,0)  

            ret = self.detector.run(image_or_path_or_tensor=img_crop,thresh=self.thresh)
            xys = ret['results'][1][0][5:]

            xys = np.array(xys,dtype=np.int32).reshape(-1,2)

            keypoints_new = [] 
            for j,e in enumerate(edges):
                if xys[e].min()>0:
                    if 32>j>=1:
                        pts = xys[e]
                        pts_ = xys[[e[1]-2]]
                                          
                        pts = np.concatenate((pts,pts_),axis=0)
                        pts = pts.reshape((-1,1,2))

                        xmin = np.min(pts[:,:,0])
                        xmax = np.max(pts[:,:,0])
                        ymin = np.min(pts[:,:,1])
                        ymax = np.max(pts[:,:,1])
                        e1_y = xys[e[1],1]
                        e2_y = xys[e[1]-2,1]
                        num_ = e1_y-ymin
                        if e2_y-ymin<num_:
                            num_ = e2_y-ymin

                        mask = np.zeros(img_crop.shape, np.uint8)
                        channel_count=3
                        ignore_mask_color = (255,)*channel_count

                        cv2.fillPoly(mask,[pts],ignore_mask_color)

                        ROI = cv2.bitwise_and(mask,img_crop) 
                        ROI[ROI==0]=255
                        ROI = ROI[ymin:ymax,xmin:xmax]
                        if ROI.shape[0]*ROI.shape[1]!=0:

                            ret = cv2.adaptiveThreshold(ROI[:,:,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,5)
                            ret = 255-ret
                            ret[ret==255] = 1

                            skeleton = morphology.skeletonize(ret)
                            skeleton = skeleton.astype('float64')

                            #get Anthode points
                            skeleton_= skeleton[:num_,:]
                            add = 0
                            if num_>=100:
                                add = num_-95
                                skeleton_= skeleton[num_-95:num_-5,:]

                            idx=np.array(np.where(skeleton_==1))  #array:[[y1,...yn],[x1,....xn]
                            x0 = []
                            y0 = []
                            for i in range(idx.shape[1]):
                                if np.sum(skeleton_[idx[0,i],:])<=1.0:
                                    x0.append(idx[0,i])
                                    y0.append(idx[1,i])

                            if len(y0)>1:
                                A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]

                                x_ = int(A1*(xys[e[1]][1]-ymin-add)+B1)
                                keypoints_new.append([xys[e[0]][0], xys[e[0]][1]])
                                keypoints_new.append([x_+xmin, xys[e[1]][1]])
                            else:
                                heritical_gradient = dst[num_+ymin,xys[e[1]-2][0]:xys[e[1]][0]].reshape(1,-1).tolist()[0]
                                if len(heritical_gradient)==0:
                                    gradients_index = len(heritical_gradient)//2
                                else:
                                    i_ = [i for i in range(len(heritical_gradient)) if np.abs(heritical_gradient[i])>80]
                                    if len(i_)>1:
                                        gradients_index = (i_[0]+i_[-1])//2
                                    else:
                                        gradients_index = len(heritical_gradient)//2
                        

                                keypoints_new.append([xys[e[0]][0],xys[e[0]][1]])
                                keypoints_new.append([xys[e[1]][0]-len(heritical_gradient)+gradients_index+1,xys[e[1]][1]])
                        else:
                            keypoints_new.append([xys[e[0]][0], xys[e[0]][1]])
                            keypoints_new.append([xys[e[1]][0], xys[e[1]][1]])                          

                                    
                    elif j==0:
                        heritical_gradient = dst[xys[e[1]][1]-10,:xys[e[1]][0]].reshape(1,-1).tolist()[0]
                        if len(heritical_gradient)==0:
                            keypoints_new.append([xys[e[0]][0],xys[e[0]][1]])
                            keypoints_new.append([xys[e[1]][0],xys[e[1]][1]])
                        else:
                            i_ = [i for i in range(len(heritical_gradient)) if np.abs(heritical_gradient[i])>80]
                            if len(i_)>1:
                                gradients_index = (i_[0]+i_[-1])//2
                            else:
                                gradients_index = len(heritical_gradient)//2

                    

                            keypoints_new.append([xys[e[0]][0],xys[e[0]][1]])
                            keypoints_new.append([xys[e[1]][0]-len(heritical_gradient)+gradients_index+1,xys[e[1]][1]])


                    elif j==32:

                        pts = xys[e] #负极和正极
                        pts_ = np.array([[xys[-1,0]+30,xys[-2,1]]]) #正极
                        pts = np.concatenate((pts,pts_),axis=0)
                        pts = pts.reshape((-1,1,2))

                        xmin = np.min(pts[:,:,0])
                        xmax = np.max(pts[:,:,0])
                        ymin = np.min(pts[:,:,1])
                        ymax = np.max(pts[:,:,1])
                        e1_y = pts[1,0,1] #正极
                        num_ = e1_y-ymin

                        mask = np.zeros(img_crop.shape, np.uint8)
                        channel_count=3
                        ignore_mask_color = (255,)*channel_count

                        cv2.fillPoly(mask,[pts],ignore_mask_color)

                        ROI = cv2.bitwise_and(mask,img_crop) 
                        ROI[ROI==0]=255
                        ROI = ROI[ymin:ymax,xmin:xmax]
                        if ROI.shape[0]*ROI.shape[1]!=0:

                            ret = cv2.adaptiveThreshold(ROI[:,:,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,5)
                            ret = 255-ret
                            ret[ret==255] = 1

                            skeleton = morphology.skeletonize(ret)
                            skeleton = skeleton.astype('float64')

                            #get Anthode points
                            skeleton_= skeleton[:num_,:]
                            add = 0
                            if num_>=100:
                                add = num_-95
                                skeleton_= skeleton[num_-95:num_-5,:]

                            idx=np.array(np.where(skeleton_==1))  #array:[[y1,...yn],[x1,....xn]
                            x0 = []
                            y0 = []
                            for i in range(idx.shape[1]):
                                if np.sum(skeleton_[idx[0,i],:])<=1.0:
                                    x0.append(idx[0,i])
                                    y0.append(idx[1,i])

                            if len(y0)>1:
                                A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]

                                x_ = int(A1*(xys[e[1]][1]-ymin-add)+B1)
                                keypoints_new.append([xys[e[0]][0], xys[e[0]][1]])
                                keypoints_new.append([x_+xmin, xys[e[1]][1]])
                            else:
                                heritical_gradient = dst[num_+ymin,xys[e[1]][0]:xys[e[0]][0]+15].reshape(1,-1).tolist()[0]
                                if len(heritical_gradient)==0:
                                    gradients_index = len(heritical_gradient)//2
                                else:
                                    i_ = [i for i in range(len(heritical_gradient)) if np.abs(heritical_gradient[i])>80]
                                    if len(i_)>1:
                                        gradients_index = (i_[0]+i_[-1])//2
                                    else:
                                        gradients_index = len(heritical_gradient)//2
                        

                                keypoints_new.append([xys[e[0]][0],xys[e[0]][1]])
                                keypoints_new.append([xys[e[1]][0]+gradients_index+1,xys[e[1]][1]])
                        else:
                            keypoints_new.append([xys[e[0]][0], xys[e[0]][1]])
                            keypoints_new.append([xys[e[1]][0], xys[e[1]][1]])
            keypoints_new = np.array(keypoints_new)
            keypoints_new[:,0] +=xmin_
            keypoints_new[:,1] +=ymin_
            xys[:,0]+=xmin_
            xys[:,1]+=ymin_

            result = self.show_duiqi(img_rotate,img_name,keypoints_new,xys,rank)
            new_dict[img_name] = result
        return new_dict

    def keypoint_confirm(self,x,y,image):
        kern = image[y:y+12,x-5:x+6,0]
        countor = kern[kern>=254].size
        return countor

    def show_duiqi(self,img_rotate,img_name,keypoints_new,keypoints,rank):
        if keypoints_new.shape[0]!=66:
            print(img_name,keypoints_new.shape)
            return {}
        else:

            img_endwith = img_name.split('-')[-1][0]
            jizu_code = img_name.split('/')[-1].split('-')[0]

            imgdate = img_name.split('-')[-2][:14]
            pixl_dis = self.conf['pixel'][img_endwith]
            test_point = self.conf['test_points'][img_endwith]
            low_limit = self.conf['low_limit']
            up_lim = self.conf['up_limit'][img_endwith]
            edge_dis = []
    
            ok=True
            dis_1 = np.sqrt((keypoints[0,0]-keypoints[2,0])**2+(keypoints[0,1]-keypoints[2,1])**2)
            if dis_1<=3:
                ok=False
                cv2.circle(img_rotate, (keypoints[0][0], keypoints[0][1]),
                            2, (0, 0, 255), -1)
            else:
                cv2.circle(img_rotate, (keypoints[0][0], keypoints[0][1]),
                            2, (0, 255, 0), -1)
            cv2.circle(img_rotate, (keypoints[1][0], keypoints[1][1]),  # plot true cathode point
                            2, (0, 255, 0), -1)
            cv2.circle(img_rotate, (keypoints_new[1][0], keypoints_new[1][1]),  # plot F(true cathode) point
                            2, (0, 140, 255), -1)
            CH=False
            BG=False
            for i in range(2,len(keypoints)-2):

                if i%2==0:
                    dis_x1 = np.sqrt((keypoints[i,0]-keypoints[i-2,0])**2+(keypoints[i,1]-keypoints[i-2,1])**2)
                    dis_x2 = np.sqrt((keypoints[i,0]-keypoints[i+2,0])**2+(keypoints[i,1]-keypoints[i+2,1])**2)
                    countor = self.keypoint_confirm(x=keypoints[i,0],y=keypoints[i,1],image=img_rotate)
                    if countor>130:
                        BG=True
                    if dis_x1<3 or dis_x2<3:
                        CH=True

                    if dis_x1<3 or dis_x2<3 or countor>130:
                        ok = False
                        cv2.circle(img_rotate, (keypoints[i][0] , keypoints[i][1]),   #plot true Anthode point
                                2, (0, 0, 255), -1)

                    else:
                        cv2.circle(img_rotate, (keypoints[i][0], keypoints[i][1] ),
                                2, (0, 255, 0), -1)
                    
                else:
                    cv2.circle(img_rotate, (keypoints[i][0], keypoints[i][1] ),   # plot true Cathode point
                                2, (0, 255, 0), -1)
                    cv2.circle(img_rotate, (keypoints_new[i][0], keypoints_new[i][1] ),   #plot F(true cathode) point
                                2, (0, 140, 255), -1)

            cv2.circle(img_rotate, (keypoints[63][0], keypoints[63][1]),
                        2, (0, 255, 0), -1)

            cv2.circle(img_rotate, (keypoints_new[63][0], keypoints_new[63][1]),
                        2, (0, 140, 255), -1)

            cv2.circle(img_rotate, (keypoints[64][0], keypoints[64][1]),
                        2, (0, 255, 0), -1)

            cv2.circle(img_rotate, (keypoints_new[65][0], keypoints_new[65][1]),
                        2, (0, 140, 255), -1)
            
            for j, e in enumerate(edges_new):
                if keypoints_new[e].min() > 0:
                    dis = int(np.sqrt((keypoints_new[e[0], 1]-keypoints_new[e[1], 1])**2+(keypoints_new[e[0], 0]-keypoints_new[e[1], 0])**2)*pixl_dis*1000)/1000.
                    edge_dis.append(dis)

                    if low_limit<=dis<=up_lim:

                        cv2.line(img_rotate, (keypoints_new[e[0], 0], keypoints_new[e[0], 1]),
                                    (keypoints_new[e[1], 0], keypoints_new[e[1], 1]), (255,0,0), 1,
                                    lineType=cv2.LINE_AA)

                        cv2.putText(img_rotate,'L%s = %s'%(j+1,dis),(10,450+30*(j+1)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                    else:
                        cv2.line(img_rotate, (keypoints_new[e[0], 0], keypoints_new[e[0], 1]),
                                    (keypoints_new[e[1], 0], keypoints_new[e[1], 1]), (0,0,255), 1,
                                    lineType=cv2.LINE_AA)
                        ok = False

                        cv2.putText(img_rotate,'L%s = %s'%(j+1,dis),(10,450+30*(j+1)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)



            AnodeDL = int((np.max(keypoints_new[::2,1])-np.min(keypoints_new[::2,1]))*pixl_dis*1000)/1000.  #遍历负极点的坐标
            CathodeDL = int((np.max(keypoints_new[1::2,1])-np.min(keypoints_new[1::2,1]))*pixl_dis*1000)/1000.  #遍历正极点的坐标

            list_An = keypoints_new[::2,1].tolist()
            max_An = list_An.index(np.min(list_An))*2
            min_An = list_An.index(np.max(list_An))*2

            list_Ca = keypoints_new[1::2,1].tolist()
            max_Ca = list_Ca.index(np.min(list_Ca))*2+1
            min_Ca = list_Ca.index(np.max(list_Ca))*2+1

            cv2.circle(img_rotate, (keypoints_new[max_An][0] , keypoints_new[max_An][1]),   #plot true Anthode point
                                4, (196, 0, 255), 0)

            cv2.circle(img_rotate, (keypoints_new[min_An][0] , keypoints_new[min_An][1]),   #plot true Anthode point
                                4, (196, 0, 255), 0)

            cv2.circle(img_rotate, (keypoints_new[max_Ca][0] , keypoints_new[max_Ca][1]),   #plot true Anthode point
                                4, (0, 255, 255), 0)
            
            cv2.circle(img_rotate, (keypoints_new[min_Ca][0] , keypoints_new[min_Ca][1]),   #plot true Anthode point
                                4, (0, 255, 255), 0)

            cv2.putText(img_rotate,'An_ymin=%s,An_ymax=%s'%(np.min(list_An),np.max(list_An)),
                                    (600,190),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,100),2)
            cv2.putText(img_rotate,'Ca_ymin=%s,Ca_ymax=%s'%(np.min(list_Ca),np.max(list_Ca)),
                                    (600,230),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,100),2)



            DL = int((np.max(keypoints_new[:,1])-np.min(keypoints_new[:,1]))*pixl_dis*1000)/1000.
            DL_usl = self.conf['DL']['threshold']
            if DL>DL_usl:
                ok = False
                cv2.putText(img_rotate,'DL[%s] = %s mm'%(DL_usl,DL),(10,320),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(img_rotate,'DL[%s] = %s mm'%(DL_usl,DL),(10,320),cv2.FONT_HERSHEY_SIMPLEX,1,(200,50,100),2)

            Anode_usl = self.conf['AnodeDL']['threshold']
            if AnodeDL>Anode_usl:
                ok=False
                An_align=True
                cv2.putText(img_rotate,'AnodeDL[%s] = %s mm'%(Anode_usl,AnodeDL),(10,350),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(img_rotate,'AnodeDL[%s] = %s mm'%(Anode_usl,AnodeDL),(10,350),cv2.FONT_HERSHEY_SIMPLEX,1,(200,50,100),2)

            Cathode_usl = self.conf['CathodeDL']['threshold']
            if CathodeDL>Cathode_usl:
                ok=False
                Ca_align=True
                cv2.putText(img_rotate,'CathodeDL[%s] = %s mm'%(Cathode_usl,CathodeDL),(10,380),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(img_rotate,'CathodeDL[%s] = %s mm'%(Cathode_usl,CathodeDL),(10,380),cv2.FONT_HERSHEY_SIMPLEX,1,(200,50,100),2)

            if rank=='B':
                cv2.putText(img_rotate,'(B)',(150,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,140,255),3)
            cv2.putText(img_rotate,jizu_code,(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,100),3)
            cv2.putText(img_rotate,test_point,(500,80),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,100),3)
            cv2.putText(img_rotate,imgdate,(500,110),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,100),3)
            cv2.putText(img_rotate,'dis/pixel = %s mm / pixel'%(pixl_dis),(10,190),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,100),2)
            cv2.putText(img_rotate,'L[i] = [%s ~ %s] [mm]'%(low_limit,up_lim),(10,230),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,100),2)

            if low_limit<=np.min(edge_dis)<=up_lim:
                cv2.putText(img_rotate,'Min = %s mm'%(np.min(edge_dis)),(10,260),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            else:
                CD=True
                cv2.putText(img_rotate,'Min = %s mm'%(np.min(edge_dis)),(10,260),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            if low_limit<=np.max(edge_dis)<=up_lim:
                cv2.putText(img_rotate,'Max = %s mm'%(np.max(edge_dis)),(10,290),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            else:
                CD=True
                cv2.putText(img_rotate,'Max = %s mm'%(np.max(edge_dis)),(10,290),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            if not ok:
                label='unalign'

                cv2.putText(img_rotate,'NG',(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
                if BG or CH:
                    label='abnormal'
            else:
                label = 'OK'
                cv2.putText(img_rotate,'OK',(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

            return [label,img_rotate,rank]

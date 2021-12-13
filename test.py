
import os
import cv2
import keypoints
from keypoints.detect import Stage2_det

images_root = 'D:/projects/svolt/xray/xray_align/images'
    # {'augsJ22301224202-OK-20210226032005706-2_8':'709,354'}
img_list = os.listdir(os.path.join(images_root,'cropimages'))

model = Stage2_det()
model.load_config(config_path=None)
rank='A'
for file_name in img_list:
    roi_dict  = {}
    img = cv2.imread(os.path.join(images_root,'cropimages',file_name))
    img0 = cv2.imread(os.path.join(images_root,'rowimages',file_name))
    values= [img,669,314,img0]
    roi_dict[file_name]=values
    result_ = model.detect(roi_dict,rank)
    # print(result_)
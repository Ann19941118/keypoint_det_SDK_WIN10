from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os

class Debugger(object):
  def __init__(self, ipynb=False, theme='black', 
               num_classes=-1, dataset=None, down_ratio=4):
    self.ipynb = ipynb
    if not self.ipynb:
      import matplotlib.pyplot as plt
      self.plt = plt
    self.imgs = {}
    self.theme = theme
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
    self.dim_scale = 1
    if dataset == 'coco':
      self.names = ['p']
      self.num_class = 1
      self.num_joints = 65
      self.edges = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], 
                    [38, 39], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49], [50, 51], [52, 53], [54, 55], [56, 57], [58, 59], [60, 61], [62, 63], [63, 64]]

      self.ec =[]
      for i in range(65):
          self.ec.append((255, 0, 0))

      self.colors_hp =[]
      for i in range(65):
          self.colors_hp .append((0, 255, 0))

    num_classes = len(self.names)
    self.down_ratio=down_ratio
    self.world_size = 64
    self.out_size = 384

  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()
    
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
    if self.theme == 'white':
      fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[img_id] = (back * (1. - trans) + fore * trans)
    self.imgs[img_id][self.imgs[img_id] > 255] = 255
    self.imgs[img_id][self.imgs[img_id] < 0] = 0
    self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

  
  def gen_colormap(self, img, output_res=None):
    img = img.copy()
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map
    
  def gen_colormap_hp(self, img, output_res=None):
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map

  def add_coco_hp(self, points, img_id='default'): 
    points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
    for j in range(self.num_joints):
      cv2.circle(self.imgs[img_id],
                 (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
    for j, e in enumerate(self.edges):
      if points[e].min() > 0:
        cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
                      (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                      lineType=cv2.LINE_AA)

  def show_all_imgs(self, image_name,pause=False,path='./cache/', time=0):
    if not self.ipynb:
      for i, v in self.imgs.items():
        print(path+image_name+'{}'.format(i)+'.jpg')
        cv2.imwrite(os.path.join(path,image_name+'_{}'.format(i)+'.jpg'), v)
        cv2.imshow('{}'.format(i), v)
      if cv2.waitKey(1 if pause else 1) == 27:
        import sys
        sys.exit(0)
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=self.plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          self.plt.imshow(v)
      self.plt.show()


color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import time
import torch

from keypoints1.models.decode import keypoint_decode
from keypoints1.models.utils import flip_tensor, flip_lr_off, flip_lr

from keypoints.utils.post_process import multi_pose_post_process

from .base_detector import BaseDetector


class KeypointDetector(BaseDetector):
    def __init__(self, opt):
        super(KeypointDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

    def process(self, images, return_time=False,thresh=0.2):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]  # 在这里将图片送入模型,并得到返回结果  image torch.Size([1, 3, 512, 512])
            output['hm'] = output['hm'].sigmoid_()     # torch.Size([1, 1, 128, 128])
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()    # torch.Size([1, 9, 128, 128])

            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()
            
            if self.opt.flip_test:
                output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
                output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
                output['hps'] = (output['hps'][0:1] + flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2

                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 if hm_hp is not None else None
                reg = reg[0:1] if reg is not None else None
                hp_offset = hp_offset[0:1] if hp_offset is not None else None

            dets = keypoint_decode(
                output['hm'], output['wh'], output['hps'],
                reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K,thresh=thresh)


        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    # 后处理, 这个函数在 base_detector.py 里用到
    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])    # [1, 100, 24]

        dets = multi_pose_post_process(dets.copy(), [meta['c']], [meta['s']], meta['out_height'], meta['out_width'])

        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 135)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            try:
                from external.nms import soft_nms_39
            except:
                print('NMS not imported! If you need it,'
                    ' do \n cd $keypoints/external \n make')
            soft_nms_39(results[1], Nt=0.1, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:135] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results,image_name):
        debugger.add_img(image, img_id='keypoints')
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_hp(bbox[5:135], img_id='keypoints')
        debugger.show_all_imgs(image_name,pause=self.pause,path=self.opt.debug_dir)

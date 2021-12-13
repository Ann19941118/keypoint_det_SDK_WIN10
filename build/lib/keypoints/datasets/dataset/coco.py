from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch.utils.data as data


class COCO(data.Dataset):
    num_classes = 1
    num_joints = 65 #TODO
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = []  #TODO

    def __init__(self, opt, split):
        super(COCO, self).__init__()
        self.edges = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33], [34, 35], [36, 37], [38, 39], [40, 41], [42, 43], [44, 45], [46, 47], [48, 49], [50, 51], [52, 53], [54, 55], [56, 57], [58, 59], [60, 61], [62, 63], [63, 64]]
        self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,65]  #TODO
        self.data_dir = os.path.join(opt.data_dir, 'images_crop')

        # self.img_dir = os.path.join(self.data_dir, 'val')
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'val.json')
            self.img_dir = os.path.join(self.data_dir, 'val')
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                '{}.json').format(split)
            self.img_dir = os.path.join(self.data_dir, 'all')
        self.max_objs = 32
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing catface {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        csv_results = []
        tag = -1
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = 1
                for dets in all_bboxes[image_id][cls_ind]:
                    bbox = dets[:4]
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = dets[4]
                    # if score > 0.2:
                    bbox_out = list(map(self._to_float, bbox))
                    keypoints = np.concatenate([
                        np.array(dets[5:135], dtype=np.float32).reshape(-1, 2),
                        np.ones((65, 1), dtype=np.float32)], axis=1).reshape(195).tolist()
                    keypoints = list(map(self._to_float, keypoints))
                    print(len(keypoints))
                    keypoints = np.array(keypoints).reshape([65, 3]).astype(np.float32)
                    keypoints = np.delete(keypoints, -1, axis=1).reshape([1, 130]).tolist()
                    csv_result = [image_id] + keypoints[0]
                    if tag != csv_result[0]:
                        csv_results.append(csv_result)

                        detection = {
                            "image_id": image_id,
                            "category_id": int(category_id),
                            "bbox": bbox_out,
                            "score": float("{:.2f}".format(score)),
                            "keypoints": keypoints
                        }
                        detections.append(detection)
                    tag = csv_result[0]

        return detections, csv_results

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        results_, csv_results = self.convert_eval_format(results)
        print(results_)
        json.dump(results_,
                  open('{}/results.json'.format(save_dir), 'w'), indent=4,cls=NpEncoder)
        print("*****************************************************")
        print(np.array(csv_results))
        print("*****************************************************")
        np.savetxt('{}/results.csv'.format(save_dir), np.array(csv_results), delimiter=",", fmt=['%s'] + ['%s'] * 130)

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(opt.save_dir, "results.json")
        # detections  = convert_eval_format(all_boxes)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        # coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        # coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
        # coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

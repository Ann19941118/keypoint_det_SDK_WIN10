from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--task', default='keypoint_det')
        self.parser.add_argument('--dataset', default='coco',
                                 help='coco')
        self.parser.add_argument('--exp_id', default='battery_keypoint')
        self.parser.add_argument('--test',action='store_true')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '0: train mode'
                                      '1: predict mode. save the final detection results')

        self.parser.add_argument('--demo', default='./data/images_crop/val/',
                                 help='path to image')
        self.parser.add_argument('--load_model', default='./best.pth',
                                 help='path to pretrained model')


        self.parser.add_argument('--resume',action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='0,1',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed') 

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', default=True, action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.1,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. dla_34')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=5e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='5,12,25,40',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=50,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and '
                                      'test on test set')

        # test
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=1,
                                 help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')

        # dataset
        self.parser.add_argument('--not_rand_crop', action='store_true',
                                 help='not use the random crop data augmentation'
                                      'from CornerNet.')
        self.parser.add_argument('--shift', type=float, default=0.0,
                                 help='when not using random crop'
                                      'apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0.0,
                                 help='when not using random crop'
                                      'apply scale augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.0,
                                 help='probability of applying flip augmentation.')
        self.parser.add_argument('--no_color_aug', action='store_true',
                                 help='not use the color augmenation '
                                      'from CornerNet')
        # keypoints_det
        self.parser.add_argument('--aug_rot', type=float, default=0,
                                 help='probability of applying '
                                      'rotation augmentation.')

        # loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1.0,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1.0,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--hp_weight', type=float, default=1.,
                                 help='loss weight for human pose offset.')
        self.parser.add_argument('--hm_hp_weight', type=float, default=1.,
                                 help='loss weight for human keypoint heatmap.')

        # keypoint_det

        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')
        self.parser.add_argument('--center_thresh', type=float, default=0.1,
                                 help='threshold for centermap.')

        self.parser.add_argument('--dense_hp', action='store_true',
                                 help='apply weighted pose regression near center '
                                      'or just apply regression on center point.')
        self.parser.add_argument('--not_hm_hp', action='store_true',
                                 help='not estimate human joint heatmap, '
                                      'directly use the joint offset from center.')
        self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                                 help='not regress local offset for '
                                      'human joint heatmaps.')
        self.parser.add_argument('--not_reg_bbox', action='store_true',
                                 help='not regression bounding box size.')

        self.parser.add_argument('--eval_oracle_hm', action='store_true',
                                 help='use ground center heatmap.')
        self.parser.add_argument('--eval_oracle_wh', action='store_true',
                                 help='use ground truth bounding box size.')
        self.parser.add_argument('--eval_oracle_offset', action='store_true',
                                 help='use ground truth local heatmap offset.')
        self.parser.add_argument('--eval_oracle_kps', action='store_true',
                                 help='use ground truth keypoints pose offset.')
        self.parser.add_argument('--eval_oracle_hmhp', action='store_true',
                                 help='use ground truth human joint heatmaps.')
        self.parser.add_argument('--eval_oracle_hp_offset', action='store_true',
                                 help='use ground truth keypoints joint local offset.')
        self.parser.add_argument('--eval_oracle_dep', action='store_true',
                                 help='use ground truth depth.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.fix_res = not opt.keep_res
        # print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset
        opt.reg_bbox = not opt.not_reg_bbox
        opt.hm_hp = not opt.not_hm_hp
        opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        # print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = '/home/think/文档/keypointdet'
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        # print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)


        if opt.task == 'keypoint_det':

            opt.flip_idx = dataset.flip_idx
            opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 130}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            if opt.hm_hp:
                opt.heads.update({'hm_hp': 65})
            if opt.reg_hp_offset:
                opt.heads.update({'hp_offset': 2})
        else:
            assert 0, 'task not defined!'
        # print('heads', opt.heads)
        return opt

    def init(self, args=''):
        default_dataset_info = {
            'keypoint_det': {
                'default_resolution': [512, 512], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                # 'mean': [0.703, 0.703, 0.703], 'std': [0.314, 0.314, 0.314],
                'dataset': 'coco', 'num_joints': 65,
                'flip_idx': []}
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.keypoint import KeypointDataset

from .dataset.coco import COCO


dataset_factory = {
  'coco': COCO
}

_sample_factory = {
  'keypoint_det': KeypointDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  

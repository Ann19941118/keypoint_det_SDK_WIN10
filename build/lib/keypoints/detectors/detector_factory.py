from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .keypoint import KeypointDetector

detector_factory = {

  'keypoint_det': KeypointDetector, 
}

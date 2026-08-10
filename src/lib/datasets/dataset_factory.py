from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet_dance_func import CTDetDataset



from .dataset.BuildFunc import BF
dataset_factory = {

  'BF':BF
}

_sample_factory = {

  'ctdet': CTDetDataset,

}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  

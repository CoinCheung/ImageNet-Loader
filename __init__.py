
import dataloader
import numpy as np


class CDataLoader(object):

    def __init__(self, imroot, annfile, batchsize, cropsize=(224, 224), shuffle=True, nchw=True, is_train=True, num_workers=4, drop_last=True):
        #  self.shuffle = shuffle
        self.dl = dataloader.CDataLoader(imroot, annfile, batchsize, cropsize, nchw, is_train, shuffle, num_workers, drop_last)

    def __iter__(self):
        self.dl.start()
        #  if self.shuffle: self.dl.shuffle()
        return self

    def __next__(self):
        if self.dl.is_end():
            raise StopIteration
        return self.dl.get_batch()

    def set_epoch(self, ep):
        self.dl.set_epoch(ep)

    def init_dist(self, rank=0, num_ranks=1):
        self.dl.init_dist(rank, num_ranks)

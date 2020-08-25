
import os.path as osp
from .dataloader import CDataLoaderNp


class CDataLoaderNp(object):

    def __init__(self, imroot, annfile, batchsize, cropsize=(224, 224), shuffle=True, nchw=True, num_workers=4, drop_last=True):
        assert osp.exists(imroot) and osp.exists(annfile), 'imroot or annfile does not exists'
        self.dl = dataloader.CDataLoaderNp(imroot, annfile, batchsize, cropsize, nchw, shuffle, num_workers, drop_last)

    def __iter__(self):
        self.dl.restart()
        return self

    def __next__(self):
        if self.dl.is_end():
            raise StopIteration
        return self.dl.next_batch()

    def set_epoch(self, ep):
        self.dl.set_epoch(ep)
        return self

    def init_dist(self, rank=0, num_ranks=1):
        self.dl.init_dist(rank, num_ranks)
        return self

    def __len__(self):
        return self.dl.get_num_batches()

    def start(self):
        self.dl.start()

    def train(self):
        self.dl.train()
        return self

    def eval(self):
        self.dl.eval()
        return self

    def set_rand_aug(self, N=2, M=9):
        self.dl.set_rand_aug(N, M)
        return self

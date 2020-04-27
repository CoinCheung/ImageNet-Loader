
from .dataloader import CDataLoader


class CDataLoader(object):

    def __init__(self, imroot, annfile, batchsize, cropsize=(224, 224), shuffle=True, nchw=True, is_train=True, num_workers=4, drop_last=True):
        #  self.shuffle = shuffle
        self.dl = dataloader.CDataLoader(imroot, annfile, batchsize, cropsize, nchw, is_train, shuffle, num_workers, drop_last)

    def __iter__(self):
        self.dl.restart()
        return self

    def __next__(self):
        if self.dl.is_end():
            raise StopIteration
        return self.dl.next_batch()

    def set_epoch(self, ep):
        self.dl.set_epoch(ep)

    def init_dist(self, rank=0, num_ranks=1):
        self.dl.init_dist(rank, num_ranks)

    def __len__(self):
        return self.dl.get_n_batches()

    def start(self):
        self.dl.start()



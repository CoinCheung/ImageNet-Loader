
import cv2
import numpy as np
import torch

import dataloader

#  print(dataloader.get_img_by_path('./example.png').shape)
print('get im')
im = dataloader.get_img_by_path('./example.png')
print('to torch')
ten = torch.from_numpy(im)
print('delete im')
del im
print('to cuda')
ten = ten.cuda()

#  print('delete im')
#  del im
#  print('delete ten')
#  del ten
#  _ = input()

#
#  dl = dataloader.CDataLoader("/data/zzy/imagenet/train/", "grpc/train.txt", 32, [224, 224], True)
#
#  batch = dl.fetch_batch()
#  print(batch.shape)

class CDataLoader(object):

    def __init__(self, imroot, annfile, batchsize, cropsize=(224, 224), shuffle=True, nchw=True, num_workers=4, drop_last=True):
        self.shuffle = shuffle
        self.dl = dataloader.CDataLoader(imroot, annfile, batchsize, cropsize, nchw, num_workers, drop_last)

    def __iter__(self):
        self.dl.restart()
        if self.shuffle: self.dl.shuffle()
        return self

    def __next__(self):
        if self.dl.is_end():
            raise StopIteration
        return self.dl.get_batch()

batchsize = 256
num_workers = 48
drop_last = False
dl = CDataLoader("/data/zzy/imagenet/train/", "./train.txt", batchsize, [224, 224], True, num_workers=num_workers, drop_last=drop_last)

for e in range(1):
    for i, batch in enumerate(dl):
        print(i, ": ", batch.shape)
        if i == 5: break

#  print(batch[0, 1, :4, :3])
#  print(batch[1, 1, :4, :3])

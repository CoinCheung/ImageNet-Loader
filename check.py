
print('import dataloader')
from cdataloader import CDataLoader
import cdataloader
print('after')
print(cdataloader.__file__)
import numpy as np
import torch

# 1. pool-4t: 9m
# 2. async-4t: 9m

#  print(dataloader.get_img_by_path('./example.png').shape)
#  print('get im')
#  im = dataloader.get_img_by_path('./example.png')
#  print('to torch')
#  ten = torch.from_numpy(im)
#  print('delete im')
#  del im
#  print('to cuda')
#  ten = ten.cuda()

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


print('create loader')
batchsize = 256
num_workers = 4
drop_last = False
is_train = False
shuffle = True
rank = 0
num_ranks = 8
dl = CDataLoader("/data/zzy/imagenet/train/", "grpc/train.txt", batchsize, [224, 224], shuffle, is_train=is_train, num_workers=num_workers, drop_last=drop_last)
dl.init_dist(rank, num_ranks)
dl.start()

print('len of dl: ', len(dl))
print('start to iter data')
for e in range(2):
    dl.set_epoch(e + 1)
    for i, batch in enumerate(dl):
        print(i, ": ", batch[0].shape, ", ", batch[1].shape)
        if i < 10:
            print(batch[1][:10])
        #  if i == 5: break

print('ds len: ', len(dl))
del dl


from cdataloader import CDataLoaderNp
import cdataloader
import numpy as np
import torch


print('create loader')
cropsize = [224, 224]
batchsize = 256
num_workers = 4
drop_last = False
shuffle = True
rank = 0
num_ranks = 8
img_root = "/data/zzy/imagenet/train/"
anno_file = "grpc/train.txt"
dl = CDataLoaderNp(img_root, anno_file, batchsize, cropsize, shuffle,
        num_workers=num_workers, drop_last=drop_last)
dl.train()
dl.init_dist(rank, num_ranks) # remove this if it is not distributed training mode
dl.start()

print('num of batch each epoch: ', len(dl))

num_epochs = 100
for e in range(num_epochs):
    dl.set_epoch(e + 1)
    for i, (ims, lbs) in enumerate(dl):
        ims = torch.from_numpy(ims).cuda()
        lbs = torch.from_numpy(lbs).cuda()
        ...
        print(ims.size())


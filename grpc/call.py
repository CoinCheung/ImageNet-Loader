
import logging
import time
import sys
sys.path.insert(0, './commpy')
import base64
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, Queue
import queue

import grpc
import numpy as np
import cv2
import torch

from commpy.interface_pb2 import IdxRequest, BatchRequest
from commpy.interface_pb2_grpc import ImageServiceStub


q = queue.Queue(2048)
mq = Queue(2048*2)
n_get = 0

impth = './example.png'

def get_image():
    # method 1: use base64
    with open(impth, 'rb') as fr:
        data = base64.b64encode(fr.read())

    # method 2: use from file, it uses less memory
    data1 = np.fromfile(impth, dtype=np.uint8).tobytes()
    print(len(data1))
    print(len(data))
    return data1


def run_get_by_idx(idx):
    with grpc.insecure_channel(
            'localhost:50001',
            options=[
                ('grpc.max_receive_message_length', 1 << 30),
                ('grpc.max_send_message_length', 1 << 30),
            ]
        ) as channel:
        stub = ImageServiceStub(channel)

        req = IdxRequest(num=idx)
        reply = stub.get_img_by_idx(req)

        res = np.frombuffer(
                reply.data, dtype=reply.dtype).reshape(reply.shape)
        res = torch.from_numpy(res).unsqueeze(0)
        mq.put(res)
        #  q.put(res)
        #  return res


def run_get_batch():
    with grpc.insecure_channel(
            'localhost:50001',
            options=[
                ('grpc.max_receive_message_length', 1 << 30),
                ('grpc.max_send_message_length', 1 << 30),
            ]
        ) as channel:
        stub = ImageServiceStub(channel)

        #  req = BatchRequest(batchsize=256)
        req = BatchRequest()
        print('before call')
        reply = stub.get_batch(req)
        print('after call')

        ims = np.frombuffer(
                reply.data, dtype=reply.dtype).reshape(reply.shape)
        lbs = np.array(reply.labels)
        print(ims.shape)
        #  print(ims[:4, :4, :4, 0].ravel())
        #  print(lbs[:4])

        #  print(np.sum(ims))

    #  for i, im in enumerate(ims):
    #      cv2.imwrite('../tmp/save_{}.jpg'.format(i), im)

#
#  def run_get_batch(stub):
#
#      #  req = BatchRequest(batchsize=256)
#      req = BatchRequest()
#      #  print('before call')
#      reply = stub.get_batch(req)
#      #  print('after call')
#
#      ims = np.frombuffer(
#              reply.data, dtype=reply.dtype).reshape(reply.shape)
#      #  lbs = np.array(reply.labels)
#      #  print(ims.shape)
#      print(ims[:4, :1, :1, 0].ravel())
#      #  print(lbs[:4])
#
#      #  print(np.sum(ims))
#
#      #  for i, im in enumerate(ims):
#      #      cv2.imwrite('../tmp/save_{}.jpg'.format(i), im)
#

#  def run_get_batch_stream(stub):
def run_get_batch_stream():
    with grpc.insecure_channel(
            'localhost:50001',
            options=[
                ('grpc.max_receive_message_length', 1 << 30),
                ('grpc.max_send_message_length', 1 << 30),
            ]
        ) as channel:
        stub = ImageServiceStub(channel)

        req = BatchRequest(stub)
        #  print('before call')
        reply = stub.get_batch_stream(req)
        #  print('after call')

        ims = []
        for res in reply:
            ims.append(np.frombuffer(res.data, dtype=res.dtype).reshape(res.shape))
    ims = torch.from_numpy(np.concatenate(ims, axis=0))
    print(ims.size())
    print(torch.sum(ims))


def mthread_call():
    batchsize = 1024
    indices = list(range(batchsize))
    executor = ThreadPoolExecutor(max_workers=64)

    ims = list(executor.map(run_get_by_idx, indices))
    ims = torch.cat(ims, dim=0)
    print(ims.size())

def funcgggg(idx):
    t = torch.randn(1, 224, 224, 3)
    mq.put(t)

def start_prefetcher():
    batchsize = 1024 * 100
    indices = list(range(1024)) * 100
    print(len(indices)//1024)
    #  executor = ThreadPoolExecutor(max_workers=64)
    #  list(executor.map(run_get_by_idx, indices))
    pool = Pool(32)
    print('pool')
    pool.map_async(run_get_by_idx, indices)
    #  pool.map_async(funcgggg, indices)
    print('async')


def get_queue_batch():
    global n_get
    batchsize = 1024
    #  ims = [q.get() for _ in range(batchsize)]
    ims = []
    for i in range(batchsize):
        n_get += 1
        ims.append(mq.get())
    ims = torch.cat(ims, dim=0)
    print(ims.size())
    print(mq.qsize())


if __name__ == "__main__":
    logging.basicConfig()
    #  with grpc.insecure_channel(
    #          'localhost:50001',
    #          options=[
    #              ('grpc.max_receive_message_length', 1 << 30),
    #              ('grpc.max_send_message_length', 1 << 30),
    #          ]
    #      ) as channel:
    #      stub = ImageServiceStub(channel)

    start_prefetcher()
    for _ in range(1300):
        t1 = time.time()
        #  run_get_batch()
        #  run_get_batch_stream()
        #  run_get_batch_stream(stub)
        #  mthread_call()
        get_queue_batch()
        t2 = time.time()
        print('time is: {:.4f}'.format(t2 - t1))
        print('n_get: {} x 1024'.format(n_get//1024))

    #  #  run_get_by_idx()
    #  for _ in range(50):
    #      run_get_batch(stub)

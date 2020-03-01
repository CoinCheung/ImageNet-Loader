
import logging
import sys
sys.path.insert(0, './commpy')
import base64

import grpc
import numpy as np
import cv2

from commpy.interface_pb2 import IdxRequest, BatchRequest
from commpy.interface_pb2_grpc import ImageServiceStub


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


def run_get_by_idx():
    with grpc.insecure_channel(
            'localhost:50001',
            options=[
                ('grpc.max_receive_message_length', 1 << 28),
                ('grpc.max_send_message_length', 1 << 28),
            ]
        ) as channel:
        stub = ImageServiceStub(channel)

        req = IdxRequest(num=345)
        reply = stub.get_img_by_idx(req)

        res = np.frombuffer(
                reply.data, dtype=reply.dtype).reshape(reply.shape)
        print(res.shape)
        print(res[:4, :4, :].ravel())


def run_get_batch():
    with grpc.insecure_channel(
            'localhost:50001',
            options=[
                ('grpc.max_receive_message_length', 1 << 28),
                ('grpc.max_send_message_length', 1 << 28),
            ]
        ) as channel:
        stub = ImageServiceStub(channel)

        req = BatchRequest(batchsize=256)
        print('after call')
        reply = stub.get_batch(req)

        ims = np.frombuffer(
                reply.data, dtype=reply.dtype).reshape(reply.shape)
        lbs = np.array(reply.labels)
        print(ims.shape)
        print(ims[:4, :4, :4, 0].ravel())
        print(lbs[:4])


if __name__ == "__main__":
    logging.basicConfig()
    #  run_get_by_idx()
    run_get_batch()

# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import interface_pb2 as interface__pb2


class ImageServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.get_img_by_idx = channel.unary_unary(
        '/comm.ImageService/get_img_by_idx',
        request_serializer=interface__pb2.IdxRequest.SerializeToString,
        response_deserializer=interface__pb2.ImgReply.FromString,
        )
    self.get_batch = channel.unary_unary(
        '/comm.ImageService/get_batch',
        request_serializer=interface__pb2.BatchRequest.SerializeToString,
        response_deserializer=interface__pb2.BatchReply.FromString,
        )
    self.get_batch_stream = channel.unary_stream(
        '/comm.ImageService/get_batch_stream',
        request_serializer=interface__pb2.BatchRequest.SerializeToString,
        response_deserializer=interface__pb2.ImgReply.FromString,
        )


class ImageServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def get_img_by_idx(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def get_batch(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def get_batch_stream(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ImageServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'get_img_by_idx': grpc.unary_unary_rpc_method_handler(
          servicer.get_img_by_idx,
          request_deserializer=interface__pb2.IdxRequest.FromString,
          response_serializer=interface__pb2.ImgReply.SerializeToString,
      ),
      'get_batch': grpc.unary_unary_rpc_method_handler(
          servicer.get_batch,
          request_deserializer=interface__pb2.BatchRequest.FromString,
          response_serializer=interface__pb2.BatchReply.SerializeToString,
      ),
      'get_batch_stream': grpc.unary_stream_rpc_method_handler(
          servicer.get_batch_stream,
          request_deserializer=interface__pb2.BatchRequest.FromString,
          response_serializer=interface__pb2.ImgReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'comm.ImageService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))

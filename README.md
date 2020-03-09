# ImageNet-Loader

This is a practice of using grpc to load and processing imagenet training images. 

Sadly, I have tried many method, but still cannot make it faster than the pytorch dataloader. It seems that the overhead of multi-process transmission and protobuf deserialization is too large to be neglected. Maybe it is not a good idea to use grpc. Hope I can find a better idea in the future.

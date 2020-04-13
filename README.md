# ImageNet-Loader

I write this because I feel so tired of the problem of shared memory when I use default pytorch dataloader. Since that shared memory problem is mainly
from the IPC between each worker which is generally slower than using thread, I decide to write this with c++ which should be faster and requires no shared memory.

This is my own implementation of the dataloader with opencv and pybind11. Though nvidia DALI is a good dataloader, it does not support random augment right now. I mean to use this in my code of training on imagenet dataset. Therefore, it currently only supports classification.

### INSTALLATION
Just follow the steps in the `INSTALL.md`, installing required packages, compile `opencv` from source and pull `pybind11` from github.


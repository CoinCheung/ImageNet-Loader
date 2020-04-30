# ImageNet-Loader

I write this because I feel so sick of the problem of shared memory when I use default pytorch dataloader. Since that shared memory problem is mainly
from the IPC between each worker which is generally slower than using thread, I decide to write this with c++ which should be faster and requires no shared memory.

This is my own implementation of the dataloader with opencv and pybind11. Though nvidia DALI is a good dataloader, it does not support random augment right now. I mean to use this in my code of training on imagenet dataset. Therefore, it currently only supports classification.


### INSTALLATION
Just follow the steps in the `INSTALL.md`, installing required packages, compile `opencv` from source and pull `pybind11` from github.

1. dependencies
These are mainly dependencies for compiling opencv from source: 
```
    $ apt install libgoogle-glog-dev libtiff-dev libgoogle-perftools-dev libzstd-dev libgtk2.0-dev libavcodec-dev libvformat-dev libjpeg-dev libjasper-dev libpcre3 ninja-build
```
Also, Please use newer version cmake(I use [cmake3.17](https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.tar.gz))

2. build 
There are two methods to do this:  
1) the first method is building with setuptools   
```
    $ git clone --depth 1 https://github.com/CoinCheung/ImageNet-Loader.git
    $ cd ImageNet-Loader && git submodule init && git submodule update --depth 1
    $ python setup.py develop
```

2) the second method is build step by step  
build opencv:   
```
    $ cd third_party/opencv
    $ mkdir -p build && cd build
    $ cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_TBB=ON -DBUILD_TBB=ON -GNinja
    $ ninja install
```
build the dataloader:  
```
    $ cd /path/to/ImageNet-Loader
    $ mkdir -p build && cd build
    $ cmake .. -GNinja
    $ ninja
```
add to `PYTHONPATH`:  
```
    export PYTHONPATH=/path/to/ImageNet-Loader:$PYTHONPATH
```

### usage
There is a simple usage example in [demo.py](), just configure the settings and use it as normal python iterator.

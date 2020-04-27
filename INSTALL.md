
Note: this is just a simple dataloader for classification, 


## install dependencies
```
    $ apt install libgoogle-glog-dev libtiff-dev libgoogle-perftools-dev libzstd-dev libgtk2.0-dev libavcodec-dev libvformat-dev libjpeg-dev libjasper-dev libpcre3 ninja-build
```
Please use newer version cmake(I use [cmake3.17](https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.tar.gz))



## change environment variables, delete miniconda items
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

## pull third parties
git clone --depth 1 --recursive https://github.com/pybind/pybind11.git third_party/pybind11
git clone --depth 1 --recursive https://github.com/opencv/opencv.git third_party/opencv

<!-- # compile opencv from source -->
<!-- cd third_party/opencv -->
<!-- mkdir -p build -->
<!-- cd build -->
<!-- cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_GENERATE_PKGCONFIG=ON -DWITH_TBB=ON -DBUILD_TBB=ON -GNinja -->
<!-- ninja install -->

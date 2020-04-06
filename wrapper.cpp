
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <glog/logging.h>

#include "src/dataloader.hpp"
#include "src/pipeline.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"


namespace py = pybind11;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::array;


py::array get_img_by_path(py::str impth) {

    vector<float> *resptr{nullptr};
    vector<int> size;
    LoadTrainImgByPath(impth, resptr, size);
    CHECK(resptr != nullptr) << "process image error\n";

    py::capsule cap = py::capsule(resptr,
        [](void *p) {delete reinterpret_cast<vector<float>*>(p);});

    py::array res = py::array(size, resptr->data(), cap);
    return res;
}

class CDataLoader {
    public:
        DataLoader dl;

        CDataLoader(string rootpth, string fname, 
                int bs, vector<int> size, bool nchw=true
                );
        py::array get_batch();
};

CDataLoader::CDataLoader(string rootpth, string fname, 
        int bs, vector<int> size, bool nchw) {
    dl.init(rootpth, fname, bs, size, nchw);
}

py::array CDataLoader::get_batch() {
    vector<float> *data{nullptr};
    vector<int> size;
    dl.get_batch(data, size);
    CHECK(data != nullptr) << "fetch data error\n";
    for (auto el: size)
        cout << el << endl;
    py::capsule cap = py::capsule(data,
        [](void *p) {delete reinterpret_cast<vector<float>*>(p);});
    py::array res = py::array(size, data->data(), cap);
    return res;
}


PYBIND11_MODULE(dataloader, m) {
    m.doc() = "load image with c++";
    m.def("get_img_by_path", &get_img_by_path, "get single image float32 array");

    py::class_<CDataLoader>(m, "CDataLoader")
        .def(py::init<string, string, int, vector<int>, bool>())
        .def("get_batch", &CDataLoader::get_batch);
}

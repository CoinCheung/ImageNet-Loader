
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

class CDataLoader: public DataLoader {
    public:
        CDataLoader(string rootpth, string fname, 
                int bs, vector<int> size, bool nchw=true, int n_workers=4
                ): DataLoader(rootpth, fname, bs, size, nchw, n_workers) {}
        py::array get_batch();
        void restart();
        void shuffle();
        bool is_end();
};


py::array CDataLoader::get_batch() {
    vector<float> *data{nullptr};
    vector<int> size;
    _get_batch(data, size);
    CHECK(data != nullptr) << "fetch data error\n";
    py::capsule cap = py::capsule(data,
        [](void *p) {delete reinterpret_cast<vector<float>*>(p);});
    py::array res = py::array(size, data->data(), cap);
    return res;
}

void CDataLoader::restart() {_restart();}

void CDataLoader::shuffle() {_shuffle();}

bool CDataLoader::is_end() {_is_end();}


PYBIND11_MODULE(dataloader, m) {
    m.doc() = "load image with c++";
    m.def("get_img_by_path", &get_img_by_path, "get single image float32 array");

    py::class_<CDataLoader>(m, "CDataLoader")
        .def(py::init<string, string, int, vector<int>, bool, int>())
        .def("get_batch", &CDataLoader::get_batch)
        .def("restart", &CDataLoader::restart)
        .def("shuffle", &CDataLoader::shuffle)
        .def("is_end", &CDataLoader::is_end);
}

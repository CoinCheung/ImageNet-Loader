
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
        [](void *p) {
        delete reinterpret_cast<vector<float>*>(p);});

    py::array res = py::array(size, resptr->data(), cap);
    return res;
}

class CDataLoader: public DataLoader {
    public:
        CDataLoader(string rootpth, string fname, 
                int bs, vector<int> size, bool nchw=true, bool train=true, 
                bool shuffle=true, int n_workers=4, bool drop_last=true
                ): DataLoader(rootpth, fname, bs, size, nchw, train, shuffle,
                    n_workers, drop_last) {}
        py::tuple get_batch();
        void start();
        void restart();
        void shuffle();
        bool is_end();
        void set_epoch(int ep);
        void init_dist(int rank, int num_ranks);
};


py::tuple CDataLoader::get_batch() {
    vector<float> *data{nullptr};
    vector<int> size;
    vector<int> *labels{nullptr};
    _get_batch(data, size, labels);
    auto t1 = std::chrono::steady_clock::now();
    CHECK((data != nullptr) && (labels != nullptr)) << "fetch data error\n";
    py::capsule cap_data = py::capsule(data,
        [](void *p) {delete reinterpret_cast<vector<float>*>(p);});
    py::array res_data = py::array(size, data->data(), cap_data);
    py::capsule cap_lb = py::capsule(labels,
        [](void *p) {delete reinterpret_cast<vector<int>*>(p);});
    py::array res_label = py::array({size[0]}, labels->data(), cap_lb);
    auto t2 = std::chrono::steady_clock::now();
    // cout << "after _get_batch_ called: "
    //     << std::chrono::duration<double, std::milli>(t2 - t1).count() << endl;
    // return res;
    return py::make_tuple(res_data, res_label);
}

void CDataLoader::start() {_start();}

void CDataLoader::restart() {_restart();}

void CDataLoader::shuffle() {_shuffle();}

bool CDataLoader::is_end() {_is_end();}

void CDataLoader::set_epoch(int ep) {_set_epoch(ep);}

void CDataLoader::init_dist(int rank, int num_ranks) {_init_dist(rank, num_ranks);}


PYBIND11_MODULE(dataloader, m) {
    m.doc() = "load image with c++";
    m.def("get_img_by_path", &get_img_by_path, "get single image float32 array");

    py::class_<CDataLoader>(m, "CDataLoader")
        .def(py::init<string, string, int, vector<int>, bool, bool, bool, int, bool>())
        .def("get_batch", &CDataLoader::get_batch)
        .def("start", &CDataLoader::start)
        .def("restart", &CDataLoader::restart)
        .def("shuffle", &CDataLoader::shuffle)
        .def("is_end", &CDataLoader::is_end)
        .def("set_epoch", &CDataLoader::set_epoch)
        .def("init_dist", &CDataLoader::init_dist);
}

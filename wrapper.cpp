
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

class CDataLoaderNp {
    public:
        CDataLoaderNp(string rootpth, string fname, 
                int bs, vector<int> size, bool nchw=true, 
                bool shuffle=true, int n_workers=4, bool drop_last=true
                ): dl(rootpth, fname, bs, size, nchw, shuffle, n_workers, drop_last) {}
        py::tuple next_batch();
        void start() {dl._start();}
        void restart() {dl._restart();}
        void shuffle() {dl._shuffle();}
        bool is_end() {return dl._is_end();}
        void set_epoch(int ep) {dl._set_epoch(ep);}
        void init_dist(int rank, int num_ranks) {dl._init_dist(rank, num_ranks);}
        int64_t get_num_batches() {return dl._get_num_batches();}
        void train() {dl.dataset._train();}
        void eval() {dl.dataset._eval();}
        void set_rand_aug(int N, int M) {dl.dataset._set_rand_aug(N, M);}

        DataLoaderNp dl;
};

// py::tuple CDataLoader::get_batch() {
//     vector<float> *data{nullptr};
//     vector<int> size;
//     vector<int64_t> *labels{nullptr};
//     _get_batch(data, size, labels);
//     auto t1 = std::chrono::steady_clock::now();
//     CHECK((data != nullptr) && (labels != nullptr)) << "fetch data error\n";
//     py::capsule cap_data = py::capsule(data,
//         [](void *p) {delete reinterpret_cast<vector<float>*>(p);});
//     py::array res_data = py::array(size, data->data(), cap_data);
//     py::capsule cap_lb = py::capsule(labels,
//         [](void *p) {delete reinterpret_cast<vector<int64_t>*>(p);});
//     py::array res_label = py::array({size[0]}, labels->data(), cap_lb);
//     auto t2 = std::chrono::steady_clock::now();
//     // cout << "after _get_batch_ called: "
//     //     << std::chrono::duration<double, std::milli>(t2 - t1).count() << endl;
//     // return res;
//     return py::make_tuple(res_data, res_label);
// }

// TODO: make convert to numpy a separate function
py::tuple CDataLoaderNp::next_batch() {
    // auto t1 = std::chrono::steady_clock::now();
    Batch spl = dl._next_batch();
    py::capsule cap_data = py::capsule(spl.data,
        [](void *p) {delete reinterpret_cast<vector<float>*>(p);});
    py::array res_data = py::array(spl.dsize, spl.data->data(), cap_data);
    py::capsule cap_lb = py::capsule(spl.labels,
        [](void *p) {delete reinterpret_cast<vector<int64_t>*>(p);});
    py::array res_label = py::array(spl.lsize, spl.labels->data(), cap_lb);
    // auto t2 = std::chrono::steady_clock::now();
    // cout << "after _get_batch_ called: "
    //     << std::chrono::duration<double, std::milli>(t2 - t1).count() << endl;
    // return res;
    return py::make_tuple(res_data, res_label);
}


PYBIND11_MODULE(dataloader, m) {
    m.doc() = "load image with c++";
    m.def("get_img_by_path", &get_img_by_path, "get single image float32 array");

    py::class_<CDataLoaderNp>(m, "CDataLoaderNp")
        .def(py::init<string, string, int, vector<int>, bool, bool, int, bool>())
        .def("next_batch", &CDataLoaderNp::next_batch)
        .def("start", &CDataLoaderNp::start)
        .def("restart", &CDataLoaderNp::restart)
        .def("shuffle", &CDataLoaderNp::shuffle)
        .def("is_end", &CDataLoaderNp::is_end)
        .def("set_epoch", &CDataLoaderNp::set_epoch)
        .def("init_dist", &CDataLoaderNp::init_dist)
        .def("get_num_batches", &CDataLoaderNp::get_num_batches)
        .def("train", &CDataLoaderNp::train)
        .def("eval", &CDataLoaderNp::eval)
        .def("set_rand_aug", &CDataLoaderNp::set_rand_aug);
}

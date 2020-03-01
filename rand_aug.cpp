
#include <vector>
#include <glog/logging.h>
#include <iterator>

#include "rand_aug.hpp"
#include "transforms.hpp"
#include "random.hpp"


using std::vector;


RandAug::RandAug(): N(0), M(9) {
    Register_ops();
}


RandAug::RandAug(int N, int M): N(N), M(M) {
    Register_ops();
}


void RandAug::Register_ops() {
    double prob = 1;
    ops.emplace_back(new Cutout("Cutout", prob, 40, {128, 128, 128}, true));
    ops.emplace_back(new Equalize("Equalize", prob));
    ops.emplace_back(new Autocontrast("Autocontrast", prob, 10));
    num_ops = 3;
}


// TODO: 
// 1. add parse magnitude logic
// 2. try to use std::bind and std::function
// 3. add operations of implement translateX and translateY
Mat RandAug::operator()(Mat& im) {
    vector<int64_t> op_idx = grandom.randint(0, num_ops, N);
    std::copy(op_idx.begin(), op_idx.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
    std::cout << std::endl;
    Mat res = im;
    for (int64_t idx : op_idx) {
        res = ops[idx]->Apply(res);
    }
    return res;
}


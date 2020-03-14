
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


// NOTES: though std::function can also work, using std::unique_ptr + emplace_pack(new ...) would be more efficient
void RandAug::Register_ops() {
    double prob = 1;
    int M = 9;
    int cutout_const = 40;
    int cutoff = 10; // autocontrast
    array<uint8_t, 3> replace{128, 128, 128};
    bool inplace = true;

    // ops.emplace_back(new Cutout("Cutout", prob, M, cutout_const, replace, inplace));
    // ops.emplace_back(new Equalize("Equalize", prob, M));
    // ops.emplace_back(new Autocontrast("Autocontrast", prob, M, cutoff));
    // ops.emplace_back(new Rotate("Rotate", prob, M, replace, inplace));
    // ops.emplace_back(new ShearX("ShearX", prob, M, replace, inplace));
    // ops.emplace_back(new ShearY("ShearY", prob, M, replace, inplace));
    ops.emplace_back(new TranslateX("TranslateX", prob, M, replace, inplace));
    ops.emplace_back(new TranslateY("TranslateY", prob, M, replace, inplace));
    ops.emplace_back(new Sharpness("Sharpness", prob, M, inplace));

    num_ops = ops.size();
}


// TODO: 
// done 1. add parse magnitude logic
// dicard for low efficiency 2. try to use std::bind and std::function
// 3. add operations of implement translateX and translateY
Mat RandAug::operator()(Mat& im) {
    vector<int64_t> op_idx = grandom.randint(0, num_ops, N);
    std::copy(op_idx.begin(), op_idx.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
    std::cout << std::endl;
    Mat res = im;
    for (int64_t idx : op_idx) {
        res = ops[idx]->FuncWithProb(res);
    }
    return res;
}


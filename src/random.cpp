
#include <random>
#include <glog/logging.h>
#include <vector>
#include <algorithm>

#include "random.hpp"


using std::vector;


Random::Random() {
    std::random_device rd;
    // engine.seed(123);
    engine.seed(rd());
}


Random::Random(int seed) {
    engine.seed(seed);
}


void Random::set_seed(int seed) {
    engine.seed(seed);
}


double Random::rand() {
    std::uniform_real_distribution<double> dist(0, 1);
    return dist(engine);
}


double Random::rand(double from, double to) {
    std::uniform_real_distribution<double> dist(from, to);
    return dist(engine);
}


double Random::normal(double mean, double std) {
    std::normal_distribution<double> dist(mean, std);
    return dist(engine);
}


vector<int64_t> Random::randint(int64_t from, int64_t to, int64_t num, bool non_repeat) {
    CHECK_LE(from,  to) << "random range error, from should less than to\n";
    vector<int64_t> res(num);
    if (from == to) {
        for (int64_t i{0}; i < num; ++i) res[i] = from;
    } else if (non_repeat) {
        int64_t n = to - from;
        CHECK_GE(n, num) << "range length must greater than num required\n";
        vector<int64_t> pool(n);
        std::iota(pool.begin(), pool.end(), from);
        std::shuffle(pool.begin(), pool.end(), engine);
        std::copy(pool.begin(), pool.begin() + num, res.begin());
    } else {
        std::uniform_int_distribution<int64_t> dist(from, to - 1);
        for (int64_t i{0}; i < num; ++i) res[i] = dist(engine);
    }
    return res;
}


int64_t Random::randint(int64_t from, int64_t to) {
    CHECK_LE(from,  to) << "random range error, from should less than to\n";
    int64_t res;
    if (from == to) {
        res = from;
    } else {
        std::uniform_int_distribution<int64_t> dist(from, to - 1);
        res = dist(engine);
    }
    return res;
}

Random grandom;

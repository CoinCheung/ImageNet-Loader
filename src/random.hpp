#ifndef _RANDOM_HPP_
#define _RANDOM_HPP_


#include <random>
#include <vector>

using std::vector;


class Random {
    public:
        // std::minstd_rand engine;
        std::mt19937 engine;

        Random();
        Random(int seed);

        // method
        void set_seed(int seed);
        double rand();
        double rand(double from, double to);
        int64_t randint(int64_t from, int64_t to);
        double normal(double mean, double std);
        vector<int64_t> randint(int64_t from, int64_t to, int64_t num, bool non_repeat=false);

};


extern Random grandom;



#endif

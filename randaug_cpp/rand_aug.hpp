
#ifndef _RAND_AUG_HPP_
#define _RAND_AUG_HPP_

#include <vector>
#include <memory>

#include "transforms.hpp"


using std::vector;
using std::unique_ptr;


class RandAug {
    public:
        vector<unique_ptr<RandApply>> ops;
        int N;
        int M;
        int num_ops;

        RandAug(); 
        RandAug(int N, int M); 

        void Register_ops();
        Mat operator()(Mat&);
};




#endif

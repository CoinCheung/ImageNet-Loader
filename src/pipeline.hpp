
#ifndef _PIPELINE_HPP_
#define _PIPELINE_HPP_

#include <array>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "rand_aug.hpp"

using std::string;
using std::array;
using std::vector;
using cv::Mat;

void LoadTrainImgByPath(string impth, vector<float>* &res, vector<int> &size, bool CHW=true);
Mat TransTrain(Mat& im, array<int, 2> size,  bool inplace=true);
void Mat2Mem (Mat &im, float* res, bool CHW);
void Mat2Vec (Mat &im, vector<float>* &res, vector<int>& size, bool CHW=true);


class DataSet {
    public: 
        array<int, 2> size;
        bool inplace;
        RandAug ra;

        DataSet(array<int, 2> size={224, 224}, int ra_n=2, int ra_m=9, bool inplace=true): size(size), inplace(inplace) {ra = RandAug(ra_n, ra_m);}

        Mat operator()(Mat& im);
        void Mat2Mem (Mat &im, float* res, bool CHW);
};


extern const int height;
extern const int width;

#endif

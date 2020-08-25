
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
        vector<string> img_paths;
        vector<int64_t> labels;
        int n_samples;
        array<int, 2> size;
        bool inplace{true};
        bool is_train{true};
        bool nchw{true};
        RandAug ra;
        bool use_ra{false};

        DataSet(string rootpth, string fname, array<int, 2> size={224, 224}, bool nchw=true, bool inplace=true); 
        DataSet() {set_default_states();}

        // void init(string rootpth, string fname, array<int, 2> size={224, 224}, bool nchw=true, int ra_n=2, int ra_m=9, bool inplace=true);
        void parse_annos(string imroot, string annfile);
        Mat TransTrain(Mat& im);
        Mat TransVal(Mat& im);
        void get_one_by_idx(int idx, float* data, int64_t& lb);
        void Mat2Mem (Mat &im, float* res);
        int get_n_samples();
        void set_default_states();

        void _train();
        void _eval();
        void _set_rand_aug(int ra_n=2, int ra_m=9);
};


extern const int height;
extern const int width;

#endif

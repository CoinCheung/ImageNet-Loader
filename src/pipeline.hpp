
#ifndef _PIPELINE_HPP_
#define _PIPELINE_HPP_

#include <array>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using std::string;
using std::array;
using std::vector;
using cv::Mat;

void LoadTrainImgByPath(string impth, vector<float>* &res, vector<int> &size, bool CHW=true);
Mat TransTrain(Mat& im, array<int, 2> size,  bool inplace=true);
void Mat2Mem (Mat &im, float* res, bool CHW);
void Mat2Vec (Mat &im, vector<float>* &res, vector<int>& size, bool CHW=true);

extern const int height;
extern const int width;

#endif

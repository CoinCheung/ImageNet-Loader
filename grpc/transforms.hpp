
#ifndef _TRANSFORMS_HPP_
#define _TRANSFORMS_HPP_

#include <array>
#include <string>
#include <opencv2/opencv.hpp>

#include "random.hpp"


using std::array;
using std::string;
using cv::Mat;


// Functions
Mat RandomHorizontalFlip(Mat &im, double p=0.5, bool inplace=true);
Mat RandomResizedCrop(Mat &im, array<int, 2> size,
        array<double, 2> scale={0.08, 1.},
        array<double, 2> ratio={3./4., 4./3.});
Mat Normalize(Mat &im, array<double, 3> mean, array<double, 3> std);
Mat HWC2CHW (Mat &im);
Mat TransTrain(Mat& im, array<int, 2> size,  bool inplace=true);


Mat EqualizeFunc(Mat &im);
Mat AutocontrastFunc(Mat &im, int cutoff=0);
Mat CutoutFunc(Mat &im, int pad_size, array<uint8_t, 3> replace, bool inplace=true);


// Objects
class RandApply {
    public: 
        double p;
        string name;
        RandApply(string name, double p): p(p), name(name) {}
        virtual ~RandApply() {}

        Mat Apply(Mat &im);
        virtual Mat Func(Mat &im) = 0;
};


class Equalize: public RandApply {
    public: 
        Equalize(string name, double p): RandApply(name, p) {}

        Mat Func(Mat &im) override;
};


class Autocontrast: public RandApply {
    public: 
        int cutoff;

        Autocontrast(string name, double p, int cutoff):
            RandApply(name, p), cutoff(cutoff) {}

        Mat Func(Mat &im) override;
};


class Cutout: public RandApply {
    public: 
        int pad_size;
        array<uint8_t, 3> replace;
        bool inplace;

        Cutout(string name, double p, int pad_size, array<uint8_t, 3> replace, 
            bool inplace): RandApply(name, p), pad_size(pad_size),
            replace(replace), inplace(inplace) {}

        Mat Func(Mat &im) override;
};


#endif

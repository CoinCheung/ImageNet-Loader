
#ifndef _TRANSFORMS_HPP_
#define _TRANSFORMS_HPP_

#include <array>
#include <string>
#include <opencv2/opencv.hpp>

#include "random.hpp"


using std::array;
using std::string;
using cv::Mat;


// Basic Trans 
Mat RandomHorizontalFlip(Mat &im, double p=0.5, bool inplace=true);
Mat RandomResizedCrop(Mat &im, array<int, 2> size,
        array<double, 2> scale={0.08, 1.},
        array<double, 2> ratio={3./4., 4./3.});
Mat Normalize(Mat &im, array<double, 3> mean, array<double, 3> std);
Mat HWC2CHW (Mat &im);
Mat TransTrain(Mat& im, array<int, 2> size,  bool inplace=true);


// Trans Funcs
Mat EqualizeFunc(Mat &im);
Mat AutocontrastFunc(Mat &im, int cutoff=0);
Mat CutoutFunc(Mat &im, int pad_size, array<uint8_t, 3> replace, bool inplace=true);


// Trans Objects
class RandApply {
    public: 
        string name;
        double p;
        int M;
        int MAX_LEVEL;

        RandApply(string name, double p, int mag): p(p), M(mag), name(name) {MAX_LEVEL = 10;}
        RandApply(string name, double p, int mag, int max_level): p(p), M(mag), name(name), MAX_LEVEL(max_level) {}
        virtual ~RandApply() {}

        Mat FuncWithProb(Mat &im); // with prob logic
        virtual Mat Func(Mat &im) = 0; // only do trans
        virtual void SetMagnitude(int mag) = 0;
};


class Equalize: public RandApply {
    public: 
        Equalize(string name, double p, int mag): RandApply(name, p, mag) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Autocontrast: public RandApply {
    public: 
        int cutoff;

        Autocontrast(string name, double p, int mag, int cutoff):
            RandApply(name, p, mag), cutoff(cutoff) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Cutout: public RandApply {
    public: 
        array<uint8_t, 3> replace;
        int cutout_const;
        bool inplace;
        int pad_size;

        Cutout(string name, double p, int mag, int cutout_const,
                array<uint8_t, 3> replace, bool inplace
                ): RandApply(name, p, mag), replace(replace),
        cutout_const(cutout_const), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


#endif

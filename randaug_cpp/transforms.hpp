
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
Mat RotateFunc(Mat &im,  float degree, array<uint8_t, 3> replace, bool inplace=true);
Mat ShearXFunc(Mat &im,  float factor, array<uint8_t, 3> replace, bool inplace=true);
Mat ShearYFunc(Mat &im,  float factor, array<uint8_t, 3> replace, bool inplace=true);
Mat TranslateXFunc(Mat &im, float offset, array<uint8_t, 3> replace, bool inplace=true);
Mat TranslateYFunc(Mat &im, float offset, array<uint8_t, 3> replace, bool inplace=true);
Mat SharpnessFunc(Mat &im, float factor, bool inplace=true);
Mat PosterizeFunc(Mat &im, int bits, bool inplace=true);
Mat ColorFunc(Mat &im, float factor, bool inplace=true);
Mat InvertFunc(Mat &im, bool inplace=true);
Mat ContrastFunc(Mat &im, float factor, bool inplace=true);
Mat BrightnessFunc(Mat &im, float factor, bool inplace=true);
Mat SolarizeFunc(Mat &im, int thresh, bool inplace=true);
Mat SolarizeAddFunc(Mat &im, int addition, int thresh, bool inplace=true);


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


class Rotate: public RandApply {
    public: 
        float degree; 
        array<uint8_t, 3> replace;
        bool inplace;

        Rotate(string name, double p, int mag,
                array<uint8_t, 3> replace, bool inplace
                ): RandApply(name, p, mag), replace(replace), 
                inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class ShearX: public RandApply {
    public: 
        float factor; 
        array<uint8_t, 3> replace;
        bool inplace;

        ShearX(string name, double p, int mag,
                array<uint8_t, 3> replace, bool inplace
                ): RandApply(name, p, mag), replace(replace), 
                inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class ShearY: public RandApply {
    public: 
        float factor; 
        array<uint8_t, 3> replace;
        bool inplace;

        ShearY(string name, double p, int mag,
                array<uint8_t, 3> replace, bool inplace
                ): RandApply(name, p, mag), replace(replace), 
                inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class TranslateX: public RandApply {
    public: 
        float offset; 
        array<uint8_t, 3> replace;
        bool inplace;

        TranslateX(string name, double p, int mag,
                array<uint8_t, 3> replace, bool inplace
                ): RandApply(name, p, mag), replace(replace), 
                inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class TranslateY: public RandApply {
    public: 
        float offset; 
        array<uint8_t, 3> replace;
        bool inplace;

        TranslateY(string name, double p, int mag,
                array<uint8_t, 3> replace, bool inplace
                ): RandApply(name, p, mag), replace(replace), 
                inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Sharpness: public RandApply {
    public: 
        float factor; 
        bool inplace;

        Sharpness(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Posterize: public RandApply {
    public: 
        int bits; 
        bool inplace;

        Posterize(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Color: public RandApply {
    public: 
        float factor; 
        bool inplace;

        Color(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Invert: public RandApply {
    public: 
        bool inplace;

        Invert(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Contrast: public RandApply {
    public: 
        float factor;
        bool inplace;

        Contrast(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Brightness: public RandApply {
    public: 
        float factor;
        bool inplace;

        Brightness(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class Solarize: public RandApply {
    public: 
        int thresh;
        bool inplace;

        Solarize(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};


class SolarizeAdd: public RandApply {
    public: 
        int addition;
        int thresh;
        bool inplace;

        SolarizeAdd(string name, double p, int mag, bool inplace
                ): RandApply(name, p, mag), thresh(128), inplace(inplace) {}

        Mat Func(Mat &im) override;
        void SetMagnitude(int mag) override;
};

#endif

#include "mobilefacenet.h"

#include "net.h"

#include <opencv2/core/core.hpp>

#include <assert.h>
#include <cmath>
#include <stdio.h>
#include <string>
#include <vector>


class Mobilefacenet::Impl
{
    public:
        explicit Impl(const std::string &model_path);

        /**
         * \brief extract face feature
         * \param [in] aligned_face aligned face data
         * \param [out] feature face feature
         * \return 0 if extract feature success
         */
        int extract_feature(const cv::Mat &aligned_face, std::vector<float> &feature);

    private:
        ncnn::Net mobilefacenet;
};

Mobilefacenet::Impl::Impl(const std::string &model_path)
{
    assert(!model_path.empty());

    mobilefacenet.opt.use_vulkan_compute = true;

    std::string param_file = "mobilefacenet.param";
    std::string model_file = "mobilefacenet.bin";
    if(model_path.back() == '/' || model_path == "\\")
    {
        param_file = model_path + param_file;
        model_file = model_path + model_file;
    }
    else
    {
        param_file = model_path + "/" + param_file;
        model_file = model_path + "/" + model_file;
    }
    mobilefacenet.load_param(param_file.c_str());
    mobilefacenet.load_model(model_file.c_str());
}

int Mobilefacenet::Impl::extract_feature(const cv::Mat &aligned_face, std::vector<float> &feature)
{
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(aligned_face.data, ncnn::Mat::PIXEL_BGR, aligned_face.cols, aligned_face.rows, 112, 112);

    //const float mean_vals[3] = {104.f, 117.f, 123.f};
    //in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = mobilefacenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc1", out);

    feature.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        feature[j] = out[j];
    }

    return 0;
}

Mobilefacenet::Mobilefacenet(const std::string &model_path)
{
    impl = new Impl(model_path);
}

Mobilefacenet::~Mobilefacenet()
{
    Impl *p = (Impl *)impl;
    delete p;
    impl = nullptr;
}

int Mobilefacenet::extract_feature(const cv::Mat &aligned_face, std::vector<float> &feature)
{
    Impl *p = (Impl *)impl;
    return p->extract_feature(aligned_face, feature);
}
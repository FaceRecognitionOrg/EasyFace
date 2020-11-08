#ifndef MOBILEFACENET_H
#define MOBILEFACENET_H

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>


class Mobilefacenet
{
    public:
        explicit Mobilefacenet(const std::string &model_path);
        ~Mobilefacenet();

        /**
         * \brief extract face feature
         * \param [in] aligned_face aligned face data
         * \param [out] feature face feature
         * \return 0 if extract feature success
         */
        int extract_feature(const cv::Mat &aligned_face, std::vector<float> &feature);

    private:
        class Impl;
        Impl *impl; 
};

#endif // MOBILEFACENET_H
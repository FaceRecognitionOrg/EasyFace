#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "common.h"

#include <vector>
#include <string>

namespace easyface
{

class FaceDetector
{
public:
    /**
    * \brief Construct detector
    * \param [in] model_path
    */
    explicit FaceDetector(const std::string &model_path);

    ~FaceDetector();
    
    /**
    * \brief detect face
    * \param [in] img input image in BGR color in HWC format
    * \return vector of detected face
    */
    std::vector<FaceObject> detect(const cv::Mat &img);

private:
    FaceDetector(const FaceDetector &) = delete;
    const FaceDetector &operator=(const FaceDetector &) = delete;

private:
    void *impl;
};

}

#endif // FACE_DETECTOR_H
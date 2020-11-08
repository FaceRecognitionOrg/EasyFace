#ifndef FACE_DETECTOR_IMPL_H
#define FACE_DETECTOR_IMPL_H

#include "common.h"

#include <vector>
#include <string>

class Retinaface;

class FaceDetectorImpl
{
public:
    /**
    * \brief Construct detector
    * \param [in] model_path
    */
    explicit FaceDetectorImpl(const std::string &model_path);

    ~FaceDetectorImpl();
    
    /**
    * \brief detect face
    * \param [in] img input image in BGR color in HWC format
    * \return vector of detected face
    */
    std::vector<FaceObject> detect(const cv::Mat &img);

private:
    FaceDetectorImpl(const FaceDetectorImpl &) = delete;
    const FaceDetectorImpl &operator=(const FaceDetectorImpl &) = delete;

private:
    Retinaface *detector;
};


#endif // FACE_DETECTOR_IMPL_H
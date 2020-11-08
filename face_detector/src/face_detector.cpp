#include "face_detector.h"
#include "face_detector_impl.h"

namespace easyface
{

FaceDetector::FaceDetector(const std::string &model_path)
{
    impl = new FaceDetectorImpl(model_path);
}

FaceDetector::~FaceDetector()
{
    FaceDetectorImpl *p = (FaceDetectorImpl *)impl;
    delete p;
}

std::vector<FaceObject> FaceDetector::detect(const cv::Mat &img)
{
    FaceDetectorImpl *p = (FaceDetectorImpl *)impl;
    return p->detect(img);
}

}

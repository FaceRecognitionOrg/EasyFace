#include "face_detector_impl.h"
#include "retinaface.h"

#include <opencv2/core/core.hpp>

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <string>
#include <vector>

FaceDetectorImpl::FaceDetectorImpl(const std::string &model_path)
{
    detector = new Retinaface(model_path);
}

FaceDetectorImpl::~FaceDetectorImpl()
{
    delete detector;
}

std::vector<FaceObject> FaceDetectorImpl::detect(const cv::Mat &img)
{
    return detector->detect(img);
}
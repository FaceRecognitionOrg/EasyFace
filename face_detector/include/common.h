#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core/core.hpp>
#include <vector>


struct FaceObject
{
    cv::Rect2f rect;  // x, y, w, h
    std::vector<cv::Point2f> landmark;
    float prob;
};

#endif // COMMON_H
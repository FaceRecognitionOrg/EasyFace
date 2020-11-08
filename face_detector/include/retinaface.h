#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <vector>
#include "common.h"


class Retinaface
{
public:
    /**
     * \brief Construct detector
     * \param [in] model_path
     */
    explicit Retinaface(const std::string &model_path);
    
    ~Retinaface();

    /**
     * \brief detect face
     * \param [in] img input image in BGR color in HWC format
     * \return vector of detected face
     */
    std::vector<FaceObject> detect(const cv::Mat &img);

private:
    Retinaface(const Retinaface &) = delete;
    const Retinaface &operator=(const Retinaface &) = delete;

private:
    class Impl;
    Impl *impl;
};


#endif // RETINAFACE_H

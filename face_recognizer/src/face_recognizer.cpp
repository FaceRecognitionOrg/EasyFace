#include "face_recognizer.h"
#include "face_recognizer_impl.h"


namespace easyface
{

FaceRecognizer::FaceRecognizer(const std::string &model_path)
{
    impl = new FaceRecognizerImpl(model_path);
}

FaceRecognizer::~FaceRecognizer()
{
    FaceRecognizerImpl *p = (FaceRecognizerImpl *)impl;
    delete p;
    impl = nullptr;
}

int FaceRecognizer::crop_face(const cv::Mat &img, const std::vector<cv::Point2f> &landmark, cv::Mat &aligned_face)
{
    FaceRecognizerImpl *p = (FaceRecognizerImpl *)impl;
    return p->crop_face(img, landmark, aligned_face);
}

int FaceRecognizer::extract_feature(const cv::Mat &aligned_face, std::vector<float> &feature)
{
    FaceRecognizerImpl *p = (FaceRecognizerImpl *)impl;
    return p->extract_feature(aligned_face, feature);
}

float FaceRecognizer::calculate_similarity(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
    FaceRecognizerImpl *p = (FaceRecognizerImpl *)impl;
    return p->calculate_similarity(feature1, feature2);
}

}
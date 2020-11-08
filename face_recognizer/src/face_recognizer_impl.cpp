#include "face_recognizer_impl.h"

#include "face_aligner.h"
#include "mobilefacenet.h"


FaceRecognizerImpl::FaceRecognizerImpl(const std::string &model_path)
{
    aligner = new FaceAligner();
    extractor = new Mobilefacenet(model_path);
}

FaceRecognizerImpl::~FaceRecognizerImpl()
{
    delete aligner;
    delete extractor;
}

int FaceRecognizerImpl::crop_face(const cv::Mat &img, const std::vector<cv::Point2f> &landmark, cv::Mat &aligned_face)
{
    return aligner->align_face(img, landmark, aligned_face);
}

int FaceRecognizerImpl::extract_feature(const cv::Mat &aligned_face, std::vector<float> &feature)
{
    return extractor->extract_feature(aligned_face, feature);
}

float FaceRecognizerImpl::calculate_similarity(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
    return calculate_cosine_similarity(feature1, feature2);
}

float FaceRecognizerImpl::calculate_cosine_similarity(const std::vector<float> &feature1, const std::vector<float> &feature2)
{
    assert(feature1.size() == feature2.size() && feature1.size() > 0);
    double dot = 0;
    double norm1 = 0;
    double norm2 = 0;
    int dim = feature1.size();
    for(int i = 0; i < dim; ++i)
    {
        dot += feature1[i] * feature2[i];
        norm1 += feature1[i] * feature1[i];
        norm2 += feature2[i] * feature2[i];
    }
    double similarity = dot / (sqrt(norm1 * norm2) + 1e-5);
    return float(similarity);
}
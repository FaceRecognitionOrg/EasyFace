#ifndef FACE_RECOGNIZER_IMPL_H
#define FACE_RECOGNIZER_IMPL_H

#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

class FaceAligner;
class Mobilefacenet;


class FaceRecognizerImpl
{
    public:
        explicit FaceRecognizerImpl(const std::string &model_path);
        ~FaceRecognizerImpl();
    
        /**
         * \brief crop face
         * \param [in] img bgr img
         * \param [in] landmark face landmark
         * \param [out] aligned_face aligned face data
         * \return 0 if crop face success
         */
         int crop_face(const cv::Mat &img, const std::vector<cv::Point2f> &landmark, cv::Mat &aligned_face);

        /**
         * \brief extract face feature
         * \param [in] aligned_face aligned face data
         * \param [out] feature face feature
         * \return 0 if extract feature success
         */
        int extract_feature(const cv::Mat &aligned_face, std::vector<float> &feature);

        /**
         * \brief calculate two faces' similarity
         * \param [in] features1 face1 features
         * \param [in] features2 face2 features
         * \return two faces' similarity
         */
        float calculate_similarity(const std::vector<float> &feature1, const std::vector<float> &feature2);

    private:
        float calculate_cosine_similarity(const std::vector<float> &feature1, const std::vector<float> &feature2);

    private:
        FaceAligner *aligner;
        Mobilefacenet *extractor;
};

#endif // FACE_RECOGNIZER_IMPL_H
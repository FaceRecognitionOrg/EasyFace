#include "face_recognizer_impl.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [modelpath] [imagepath]\n", argv[0]);
        return -1;
    }

    const char* modelpath = argv[1];
    const char* imagepath = argv[2];

    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    FaceRecognizerImpl recognizer(modelpath);
    cv::Mat aligned_face;
    std::vector<cv::Point2f> landmark = {
                                            cv::Point2f(290, 196), 
                                            cv::Point2f(383, 173), 
                                            cv::Point2f(350, 244), 
                                            cv::Point2f(325, 304), 
                                            cv::Point2f(395, 286)
                                        };                   
    std::vector<float> feature;
    recognizer.crop_face(img, landmark, aligned_face);
    recognizer.extract_feature(aligned_face, feature);
    
    // float calculate_similarity(const std::vector<float> &feature1, const std::vector<float> &feature2);

	for(int i = 0; i < feature.size(); ++i)
	{
		printf("\%f ", feature[i]);
	}
	
    return 0;
}
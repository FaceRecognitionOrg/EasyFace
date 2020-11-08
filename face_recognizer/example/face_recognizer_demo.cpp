#include "face_recognizer.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32


#ifdef _WIN32
// 单位ms
double get_current_time()
{
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
}
#else  // _WIN32
// 单位ms
double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif // _WIN32


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

    easyface::FaceRecognizer recognizer(modelpath);
    cv::Mat aligned_face;
    std::vector<cv::Point2f> landmark = {
                                            cv::Point2f(290, 196), 
                                            cv::Point2f(383, 173), 
                                            cv::Point2f(350, 244), 
                                            cv::Point2f(325, 304), 
                                            cv::Point2f(395, 286)
                                        };                   
    std::vector<float> feature;
    
    double start = get_current_time();
    recognizer.crop_face(img, landmark, aligned_face);
    double end = get_current_time();
    double time = end - start;
    fprintf(stderr, "time of crop_face is %.2f ms\n", time);

    start = get_current_time();
    recognizer.extract_feature(aligned_face, feature);
    end = get_current_time();
    time = end - start;
    fprintf(stderr, "time of extract_feature is %.2f ms\n", time);
    
    start = get_current_time();
    recognizer.calculate_similarity(feature, feature);
    end = get_current_time();
    time = end - start;
    fprintf(stderr, "time of calculate_similarity is %.2f ms\n", time);

    printf("face feature:\n");
	for(int i = 0; i < feature.size(); ++i)
	{
		printf("\%f ", feature[i]);
	}
    printf("\n");
	
    return 0;
}
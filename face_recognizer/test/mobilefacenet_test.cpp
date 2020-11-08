#include "mobilefacenet.h"

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

    cv::Mat aligned_face = cv::imread(imagepath, 1);
    if (aligned_face.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> feature;
    
    Mobilefacenet extractor(modelpath);
    extractor.extract_feature(aligned_face, feature);

	for(int i = 0; i < feature.size(); ++i)
	{
		printf("\%f ", feature[i]);
	}
	
    return 0;
}
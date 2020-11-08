/*
// size = 112 x 96
[[30.2946, 51.6963],
[65.5318, 51.5014],
[48.0252, 71.7366],
[33.5493, 92.3655],
[62.7299, 92.2041]]

// size = 112 x 112
src:  
[[38.2946   51.6963  ]
 [73.5318   51.5014  ]
 [56.0252   71.7366  ]
 [41.5493   92.3655  ]
 [70.729904 92.2041  ]]
 

landmark:
[[290 196]
[383 173]
[350 244]
[325 304]
[395 286]]
 
img_src是包含人脸的完整图片

std::vector<cv::Point2f>& keypoints = {}
*/

#include "face_aligner.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        printf("usage: %s [imagepath]", argv[0]);
        return -1;
    }
	
	const char* filename = argv[1];
    
	cv::Mat img_src = cv::imread(filename);
	std::vector<cv::Point2f> landmark = {cv::Point2f(290, 196), cv::Point2f(383, 173), cv::Point2f(350, 244), cv::Point2f(325, 304), cv::Point2f(395, 286)};
	cv::Mat aligned_face;
	FaceAligner aligner;
	aligner.align_face(img_src, landmark, aligned_face);

	cv::imwrite("aligned_face.jpg", aligned_face);
}

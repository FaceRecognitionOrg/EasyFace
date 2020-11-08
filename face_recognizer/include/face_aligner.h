#ifndef FACE_ALIGNER_H
#define FACE_ALIGNER_H

#include <opencv2/core/core.hpp>
#include <vector>

class FaceAligner 
{
	public:
		FaceAligner();
		~FaceAligner();

		int align_face(const cv::Mat & img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat &face_aligned);

	private:
		class Impl;
		Impl *impl;
};

#endif // FACE_ALIGNER_H
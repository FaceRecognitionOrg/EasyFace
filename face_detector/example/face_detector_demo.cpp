#include "face_detector.h"

#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <float.h>


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

void draw(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects, const std::string &out_filename)
{
    cv::Mat image = bgr.clone();

    for (int i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

        fprintf(stderr, "faceobject[%d]: prob = %.2f, (x, y, w, h) = (%.2f, %.2f, %.2f, %.2f)\n", i, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));

        for(int i = 0; i < obj.landmark.size(); ++i)
        {
            cv::circle(image, obj.landmark[i], 2, cv::Scalar(0, 255, 255), -1);
        }

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    //cv::imshow("image", image);
    //cv::waitKey(0);
    cv::imwrite(out_filename, image);
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [modelpath] [imagepath] [output]\n", argv[0]);
        return -1;
    }
    
    std::string modelpath = argv[1];
    std::string imagepath = argv[2];
    std::string out_filename = argv[3];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath.c_str());
        return -1;
    }

    easyface::FaceDetector detector(modelpath);

    double start = get_current_time();
    std::vector<FaceObject> faceobjects = detector.detect(m);
    double end = get_current_time();
    double time = end - start;
    fprintf(stderr, "time of face detect is %.2f ms\n", time);

    draw(m, faceobjects, out_filename);

    return 0;
}
#include "face_detector.h"
#include "face_recognizer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <float.h>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <utility>

void face_register(easyface::FaceDetector *detector, easyface::FaceRecognizer *recognizer,
    const std::string &register_imagepath, std::map<std::string, std::vector<float>> &username_feature_map)
{
    std::string pattern;
    if(register_imagepath.back() == '/' || register_imagepath.back() == '\\')
    {
        pattern = register_imagepath + "*.*";
    }
    else
    {
        pattern = register_imagepath + "/" + "*.*";
    }

    std::vector<cv::String> register_filenames;  
    cv::glob(pattern, register_filenames);

    std::cout << "Start face register... " << std::endl;
    for(auto filename : register_filenames)
    {
        std::cout << "filename: " << filename << std::endl;

        int pos = pattern.size() - 3;
        int len = filename.size() - pos - 4; // .jpg|.bmp|.png
        std::string username = filename.substr(pos, len);
        std::cout << "username: " << username << std::endl;

        cv::Mat img = cv::imread(filename, 1);
        std::vector<FaceObject> faceobjects = detector->detect(img);

        cv::Mat aligned_face;
        std::vector<cv::Point2f> landmark = faceobjects[0].landmark;                
        std::vector<float> feature;
        recognizer->crop_face(img, landmark, aligned_face);
        recognizer->extract_feature(aligned_face, feature);
        
        username_feature_map.insert(std::make_pair(username, feature));
    }

    std::cout << "Finish face register. " << std::endl;
}

void face_identify(easyface::FaceDetector *detector, easyface::FaceRecognizer *recognizer,
    const std::string &identify_filename, const std::map<std::string, std::vector<float>> &username_feature_map)
{
    std::cout << "Start face identify " << std::endl;

    int str_len = identify_filename.size(); 
    assert(str_len != 0);

    cv::VideoCapture cap;
    if(str_len == 1)
    {
        int cam_id = atoi(identify_filename.c_str());
        cap.open(cam_id);
    }
    else
    {
        cap.open(identify_filename);
    }
    if(!cap.isOpened())
    {
        std::cerr << "ERROR! Unable to open camera\n";
        return;
    }

    cv::Mat img;
    for(;;)
    {
        cap.read(img);
        if(img.empty())
        {
            std::cout << "blank frame grabbed\n";
            break;
        }

        // identify
        std::vector<FaceObject> faceobjects = detector->detect(img);

        for(const auto &obj : faceobjects)
        {
            cv::Mat aligned_face;
            std::vector<cv::Point2f> landmark = obj.landmark;                
            std::vector<float> feature;
            recognizer->crop_face(img, landmark, aligned_face);
            recognizer->extract_feature(aligned_face, feature);

            std::string name;
            float max_similarity = -FLT_MAX;
            for(const auto& item : username_feature_map)
            {
                std::string username = item.first;
                std::vector<float> register_feature = item.second;
                float similarity = recognizer->calculate_similarity(feature, register_feature);
                if(similarity > max_similarity)
                {
                    max_similarity = similarity;
                    name = username;
                }
            }

            float threshold = 0.5;
            if(max_similarity < threshold)
            {
                name = "unknown";
            }

            char text[256];
            sprintf(text, "%s, %.2f", name.c_str(), max_similarity);
            printf("identify result: %s\n", text);

            double fontscale = 1;
            int thickness = 2;
            int baseLine = 2;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontscale, thickness, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > img.cols)
                x = img.cols - label_size.width;

            cv::rectangle(img, cv::Rect(cv::Point(obj.rect.x, obj.rect.y), cv::Size(obj.rect.width, obj.rect.height)),
                      cv::Scalar(0, 255, 0), 1);

            cv::rectangle(img, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

            cv::putText(img, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, fontscale, cv::Scalar(0, 255, 0), thickness);

        }
                    
        std::string out_filename = identify_filename.substr(0, str_len - 4) + ".result.jpg";
        printf("save result to: %s\n", out_filename.c_str());
        cv::imwrite(out_filename, img);
        // show live and wait for a key with timeout long enough to show images
        /*
        imshow("face identify", img);
        char key = cv::waitKey(1);
        if (key == 27)
            break;
        */
    }

    std::cout << "Finish face identify. " << std::endl;
}

int main(int argc, char *argv[])
{
    if(argc != 5)
    {
        printf("usage: %s [detector_modelpath] [recognizer_modelpath] [register_imagepath] [identify_filename|videopath|camera_id]\n", argv[0]);
        return -1;
    }

    std::string detector_modelpath = argv[1];
    std::string recognizer_modelpath = argv[2];
    std::string register_imagepath = argv[3];
    std::string identify_filename = argv[4];

    easyface::FaceDetector detector(detector_modelpath);
    easyface::FaceRecognizer recognizer(recognizer_modelpath);
    
    std::map<std::string, std::vector<float>> username_feature_map;
    face_register(&detector, &recognizer, register_imagepath, username_feature_map);
    face_identify(&detector, &recognizer, identify_filename, username_feature_map);
    return 0;
}
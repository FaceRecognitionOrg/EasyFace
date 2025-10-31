#include "retinaface.h"

#include "net.h"

#include <opencv2/core/core.hpp>

#include <assert.h>
#include <float.h>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <string>
#include <vector>

// ref: ncnn example's retinaface.cpp

class Retinaface::Impl
{
    public:
        /**
         * \brief Construct detector
         * \param [in] model_path
         */
        Impl(const std::string &model_path);

        /**
         * \brief detect face
         * \param [in] img input image in BGR color in HWC format
         * \return vector of detected face
         */
        std::vector<FaceObject> detect(const cv::Mat &img);

    private:
        struct _FaceObject
        {
            cv::Rect_<float> rect;
            cv::Point2f landmark[5];
            float prob;
        };

    private:
        static inline float intersection_area(const _FaceObject& a, const _FaceObject& b);
        static void qsort_descent_inplace(std::vector<_FaceObject>& faceobjects);
        static void nms_sorted_bboxes(const std::vector<_FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold);
        static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales);
        static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<_FaceObject>& faceobjects);

    private:
        ncnn::Net retinaface;
        const float prob_threshold = 0.8f;
        const float nms_threshold = 0.4f;

};


Retinaface::Impl::Impl(const std::string &model_path)
{
    retinaface.opt.use_vulkan_compute = true;

    // model is converted from
    // https://github.com/deepinsight/insightface/tree/master/RetinaFace#retinaface-pretrained-models
    // https://github.com/deepinsight/insightface/issues/669
    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    //     retinaface.load_param("retinaface-R50.param");
    //     retinaface.load_model("retinaface-R50.bin");
    assert(!model_path.empty());

    std::string param_file = "mnet.25-opt.param";
    std::string model_file = "mnet.25-opt.bin";

    if(model_path.back() == '/' || model_path.back() == '\\')
    {
        param_file = model_path + param_file;
        model_file = model_path + model_file;
    }
    else
    {
        param_file = model_path + "/" + param_file;
        model_file = model_path + "/" + model_file;
    }

    retinaface.load_param(param_file.c_str());
    retinaface.load_model(model_file.c_str());

}

std::vector<FaceObject> Retinaface::Impl::detect(const cv::Mat &bgr)
{
    std::vector<FaceObject> faceobjects;

    std::vector<_FaceObject> _faceobjects;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h);

    ncnn::Extractor ex = retinaface.create_extractor();

    ex.input("data", in);

    std::vector<_FaceObject> faceproposals;

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<_FaceObject> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<_FaceObject> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector<_FaceObject> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();

    _faceobjects.resize(face_count);
    for (int i = 0; i < face_count; i++)
    {
        _faceobjects[i] = faceproposals[picked[i]];

        // clip to image size
        float x0 = _faceobjects[i].rect.x;
        float y0 = _faceobjects[i].rect.y;
        float x1 = x0 + _faceobjects[i].rect.width;
        float y1 = y0 + _faceobjects[i].rect.height;

        x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);

        _faceobjects[i].rect.x = x0;
        _faceobjects[i].rect.y = y0;
        _faceobjects[i].rect.width = x1 - x0;
        _faceobjects[i].rect.height = y1 - y0;
    }

    faceobjects.reserve(_faceobjects.size());
    for(const auto& _faceobject : _faceobjects)
    {
        FaceObject& faceobject = faceobjects.emplace_back();
        faceobject.rect = _faceobject.rect;
        faceobject.prob = _faceobject.prob;
        faceobject.landmark.resize(5);
        for(int i = 0; i < 5; ++i)
        {
            faceobject.landmark[i] = _faceobject.landmark[i];
        }
    }

    return faceobjects;
}

float Retinaface::Impl::intersection_area(const _FaceObject& a, const _FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


void Retinaface::Impl::qsort_descent_inplace(std::vector<_FaceObject>& faceobjects)
{
    if (faceobjects.size() <= 1)
        return;

    std::sort(faceobjects.begin(), faceobjects.end(), [](const _FaceObject& lhs, const _FaceObject& rhs)
    {
        return lhs.prob > rhs.prob;
    });
}


void Retinaface::Impl::nms_sorted_bboxes(const std::vector<_FaceObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();
    picked.reserve(n);

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const _FaceObject& a = faceobjects[i];

        bool keep = true;
        for (int idx : picked)
        {
            const _FaceObject& b = faceobjects[idx];

            const float inter_area = intersection_area(a, b);
            if (inter_area <= 0.f)
                continue;

            const float union_area = areas[i] + areas[idx] - inter_area;
            if (union_area <= 0.f)
                continue;

            if (inter_area / union_area > nms_threshold)
            {
                keep = false;
                break;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

// copy from src/layer/proposal.cpp
ncnn::Mat Retinaface::Impl::generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

void Retinaface::Impl::generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<_FaceObject>& faceobjects)
{
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    const size_t max_proposals = static_cast<size_t>(num_anchors) * w * h;
    if (max_proposals > 0)
    {
        faceobjects.reserve(faceobjects.size() + max_proposals);
    }

    for (int q = 0; q < num_anchors; q++)
    {
        const float* anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

        // shifted anchor
        const float anchor_w = anchor[2] - anchor[0];
        const float anchor_h = anchor[3] - anchor[1];
        const float anchor_half_w = anchor_w * 0.5f;
        const float anchor_half_h = anchor_h * 0.5f;
        const float anchor_w_plus_one = anchor_w + 1.f;
        const float anchor_h_plus_one = anchor_h + 1.f;

        const float* score_ptr = score.channel(0);
        const float* dx_ptr = bbox.channel(0);
        const float* dy_ptr = bbox.channel(1);
        const float* dw_ptr = bbox.channel(2);
        const float* dh_ptr = bbox.channel(3);
        const float* l0x_ptr = landmark.channel(0);
        const float* l0y_ptr = landmark.channel(1);
        const float* l1x_ptr = landmark.channel(2);
        const float* l1y_ptr = landmark.channel(3);
        const float* l2x_ptr = landmark.channel(4);
        const float* l2y_ptr = landmark.channel(5);
        const float* l3x_ptr = landmark.channel(6);
        const float* l3y_ptr = landmark.channel(7);
        const float* l4x_ptr = landmark.channel(8);
        const float* l4y_ptr = landmark.channel(9);

        for (int i = 0; i < h; i++)
        {
            const float anchor_y = anchor[1] + feat_stride * i;
            const float cy = anchor_y + anchor_half_h;

            float anchor_x = anchor[0];
            for (int j = 0; j < w; j++)
            {
                const float prob = *score_ptr++;

                if (prob >= prob_threshold)
                {
                    const float cx = anchor_x + anchor_half_w;
                    const float dx = *dx_ptr;
                    const float dy = *dy_ptr;
                    const float dw = *dw_ptr;
                    const float dh = *dh_ptr;

                    const float pb_cx = cx + anchor_w * dx;
                    const float pb_cy = cy + anchor_h * dy;
                    const float pb_w = anchor_w * std::exp(dw);
                    const float pb_h = anchor_h * std::exp(dh);

                    const float x0 = pb_cx - pb_w * 0.5f;
                    const float y0 = pb_cy - pb_h * 0.5f;
                    const float x1 = pb_cx + pb_w * 0.5f;
                    const float y1 = pb_cy + pb_h * 0.5f;

                    _FaceObject& obj = faceobjects.emplace_back();
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;
                    obj.landmark[0].x = cx + anchor_w_plus_one * (*l0x_ptr);
                    obj.landmark[0].y = cy + anchor_h_plus_one * (*l0y_ptr);
                    obj.landmark[1].x = cx + anchor_w_plus_one * (*l1x_ptr);
                    obj.landmark[1].y = cy + anchor_h_plus_one * (*l1y_ptr);
                    obj.landmark[2].x = cx + anchor_w_plus_one * (*l2x_ptr);
                    obj.landmark[2].y = cy + anchor_h_plus_one * (*l2y_ptr);
                    obj.landmark[3].x = cx + anchor_w_plus_one * (*l3x_ptr);
                    obj.landmark[3].y = cy + anchor_h_plus_one * (*l3y_ptr);
                    obj.landmark[4].x = cx + anchor_w_plus_one * (*l4x_ptr);
                    obj.landmark[4].y = cy + anchor_h_plus_one * (*l4y_ptr);
                    obj.prob = prob;
                }

                ++dx_ptr;
                ++dy_ptr;
                ++dw_ptr;
                ++dh_ptr;
                ++l0x_ptr;
                ++l0y_ptr;
                ++l1x_ptr;
                ++l1y_ptr;
                ++l2x_ptr;
                ++l2y_ptr;
                ++l3x_ptr;
                ++l3y_ptr;
                ++l4x_ptr;
                ++l4y_ptr;

                anchor_x += feat_stride;
            }
        }
    }
}

Retinaface::Retinaface(const std::string &model_path) : impl(new Retinaface::Impl(model_path))
{
    
}

Retinaface::~Retinaface()
{
    delete impl;
}

std::vector<FaceObject> Retinaface::detect(const cv::Mat &img)
{
    return impl->detect(img);
}
#pragma once

#include <opencv2/opencv.hpp>

inline cv::Rect getAbsoluteBbox(const cv::Rect2f &rel_bbox, cv::Size size)
{
    return cv::Rect(
        static_cast<int>(rel_bbox.x * size.width),
        static_cast<int>(rel_bbox.y * size.height),
        static_cast<int>(rel_bbox.width * size.width),
        static_cast<int>(rel_bbox.height * size.height));
}

inline cv::Mat getAbsoluteMask(const cv::Mat &rel_mask, cv::Size size, float threshold = 0.5)
{
    if (rel_mask.empty())
        return cv::Mat();

    cv::Mat resized_mask, binary_mask;
    cv::resize(rel_mask, resized_mask, size, cv::INTER_LINEAR);
    cv::threshold(resized_mask, binary_mask, threshold, 1.0, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U); // Convert to 8-bit unsigned
    return binary_mask;
}

inline cv::Mat letterbox(const cv::Mat &input, cv::Size new_shape, cv::Scalar color, bool auto_size, bool scale_fill, bool scaleup, int stride)
{
    // Resize and pad image while meeting stride-multiple constraints
    cv::Mat out(new_shape, CV_8UC3, color);
    cv::Size shape = input.size(); // current shape [height, width]

    // Scale ratio (new / old)
    float r = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);
    if (!scaleup)
    {
        r = std::min(r, 1.0f);
    }

    // Compute padding
    std::pair<float, float> ratio(r, r); // width, height ratios
    cv::Size new_unpad(static_cast<int>(std::round(shape.width * r)), static_cast<int>(std::round(shape.height * r)));
    float dw = static_cast<float>(new_shape.width - new_unpad.width);
    float dh = static_cast<float>(new_shape.height - new_unpad.height); // wh padding

    if (auto_size)
    {
        dw = std::fmod(dw, static_cast<float>(stride));
        dh = std::fmod(dh, static_cast<float>(stride)); // wh padding
    }
    else if (scale_fill)
    {
        dw = 0.0;
        dh = 0.0;
        new_unpad = new_shape;
        ratio = std::make_pair(static_cast<float>(new_shape.width) / shape.width, static_cast<float>(new_shape.height) / shape.height); // width, height ratios
    }

    dw /= 2; // divide padding into 2 sides
    dh /= 2;

    if (shape != new_unpad)
    {
        cv::resize(input, out, new_unpad, 0, 0, cv::INTER_LINEAR);
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::copyMakeBorder(out, out, top, bottom, left, right, cv::BORDER_CONSTANT, color); // add border

    return out;
}
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "detection.hpp"

struct Frame
{
    cv::Mat image;
    cv::Size size;
    std::vector<Detection> detections{};

    int width() const { return size.width; }
    int height() const { return size.height; }
};
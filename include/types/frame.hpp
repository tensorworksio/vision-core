#pragma once

#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

using TimePoint = std::chrono::system_clock::time_point;

struct Frame
{
    cv::Mat image;
    cv::Size size;
    TimePoint timestamp;

    explicit Frame(const cv::Mat &img, TimePoint ts = std::chrono::system_clock::now())
        : image(img), size(img.size()), timestamp(ts) {}

    TimePoint getTimestamp() const { return timestamp; }

    int64_t getTimestampMs() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   timestamp.time_since_epoch())
            .count();
    }

    int width() const { return size.width; };
    int height() const { return size.height; };
};
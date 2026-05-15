#pragma once

#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <types/detection.hpp>
#include <utils/detection_utils.hpp>

using TimePoint = std::chrono::system_clock::time_point;

struct Frame
{
    cv::Mat image;
    cv::Size size;
    int type{CV_8UC3};
    TimePoint timestamp;
    int64_t id;
    std::vector<Detection> detections;

    static int64_t frame_counter;

    Frame() : image(), size(0, 0), type(CV_8UC3), timestamp(std::chrono::system_clock::now()), id(frame_counter++) {}

    Frame(const cv::Mat &img, TimePoint ts = std::chrono::system_clock::now())
        : image(img), size(img.size()), type(img.type()), timestamp(ts), id(frame_counter++) {}

    friend Frame &operator>>(cv::VideoCapture &cap, Frame &frame)
    {
        cv::Mat img;
        cap >> img;
        if (!img.empty())
        {
            frame.image = img;
            frame.size = img.size();
            frame.type = img.type();
            frame.timestamp = std::chrono::system_clock::now();
            frame.id = frame_counter++;
        }
        return frame;
    }

    cv::Mat operator()(const cv::Rect &rect) const
    {
        cv::Rect safe_rect = rect & cv::Rect(0, 0, width(), height());
        return image(safe_rect);
    }

    cv::Mat operator()(const cv::Rect2f &rel_rect) const
    {
        cv::Rect safe_rect = getAbsoluteBbox(rel_rect, size);
        return image(safe_rect);
    }

    bool empty() const { return image.empty(); }

    TimePoint getTimestamp() const { return timestamp; }

    int64_t getTimestampMs() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   timestamp.time_since_epoch())
            .count();
    }

    int width() const { return size.width; };
    int height() const { return size.height; };

    int64_t getId() const { return id; }
};

int64_t Frame::frame_counter = 0;

template <>
struct rfl::Reflector<Frame>
{
    struct ReflType
    {
        int64_t id{0};
        std::chrono::system_clock::time_point timestamp{};
        cv::Size size{};
        int type{CV_8UC3};
        std::vector<Detection> detections{};
    };

    static ReflType from(const Frame &f) noexcept
    {
        return {f.id, f.timestamp, f.size, f.type, f.detections};
    }

    static Frame to(const ReflType &r) noexcept
    {
        Frame f;
        f.id = r.id;
        f.timestamp = r.timestamp;
        f.size = r.size;
        f.type = r.type;
        f.detections = r.detections;
        return f;
    }
};
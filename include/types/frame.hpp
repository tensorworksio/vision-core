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
    TimePoint timestamp;
    int64_t id;

    static int64_t frame_counter;

    Frame() : image(), size(0, 0), timestamp(std::chrono::system_clock::now()), id(frame_counter++) {}

    Frame(const cv::Mat &img, TimePoint ts = std::chrono::system_clock::now())
        : image(img), size(img.size()), timestamp(ts), id(frame_counter++) {}

    friend Frame &operator>>(cv::VideoCapture &cap, Frame &frame)
    {
        cv::Mat img;
        cap >> img;
        if (!img.empty())
        {
            frame.image = img;
            frame.size = img.size();
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

    cv::Mat draw(const std::vector<Detection> &detections, bool use_track_colors = false, bool draw_labels = true) const
    {
        cv::Mat output = image.clone();
        cv::Mat mask_overlay = cv::Mat::zeros(size, CV_8UC3);

        for (const auto &det : detections)
        {
            // Get appropriate color based on user preference
            cv::Scalar color = use_track_colors ? det.getTrackColor() : det.getClassColor();

            // Convert relative coordinates to absolute
            cv::Rect abs_bbox = det.size.empty() ? getAbsoluteBbox(det.bbox, size) : cv::Rect(det.bbox);
            cv::Mat abs_mask = getAbsoluteMask(det.mask, abs_bbox.size());

            // Draw mask if available
            if (!det.mask.empty())
            {
                cv::Mat color_mask(abs_bbox.size(), CV_8UC3, color);
                cv::Mat roi_mask = mask_overlay(abs_bbox);
                color_mask.copyTo(roi_mask, abs_mask);
            }

            // Draw bounding box
            cv::rectangle(output, abs_bbox, color, 2);

            // Draw label if requested
            if (draw_labels)
            {
                // Construct label text
                std::string label = det.class_name;
                if (det.track_id >= 0)
                {
                    label += " [" + std::to_string(det.track_id) + "]";
                }
                if (det.confidence > 0)
                {
                    label += " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
                }

                // Draw text with background
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
                cv::Point text_origin(abs_bbox.x, abs_bbox.y - 5);

                cv::rectangle(output,
                              cv::Point(text_origin.x, text_origin.y - text_size.height),
                              cv::Point(text_origin.x + text_size.width, text_origin.y + baseline),
                              color, -1);

                cv::putText(output, label, text_origin,
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
            }
        }

        // Apply the final mask overlay
        if (!mask_overlay.empty())
        {
            cv::addWeighted(output, 0.9, mask_overlay, 0.3, 0, output);
        }

        return output;
    }
};

int64_t Frame::frame_counter = 0;
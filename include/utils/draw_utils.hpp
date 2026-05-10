#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include <types/frame.hpp>
#include <utils/detection_utils.hpp>

inline cv::Mat drawDetections(const Frame &frame,
                              const std::vector<Detection> &dets,
                              bool use_track_colors = false,
                              bool draw_labels = true)
{
    cv::Mat output = frame.image.clone();
    cv::Mat mask_overlay = cv::Mat::zeros(frame.size, CV_8UC3);

    for (const auto &det : dets)
    {
        cv::Scalar color = use_track_colors ? det.getTrackColor() : det.getClassColor();

        cv::Rect abs_bbox = det.size.empty() ? getAbsoluteBbox(det.bbox, frame.size) : cv::Rect(det.bbox);
        cv::Mat abs_mask = getAbsoluteMask(det.mask, abs_bbox.size());

        if (!det.mask.empty())
        {
            cv::Mat color_mask(abs_bbox.size(), CV_8UC3, color);
            cv::Mat roi_mask = mask_overlay(abs_bbox);
            color_mask.copyTo(roi_mask, abs_mask);
        }

        cv::rectangle(output, abs_bbox, color, 2);

        if (draw_labels)
        {
            std::string label = det.class_name;
            if (det.track_id >= 0)
                label += " [" + std::to_string(det.track_id) + "]";
            if (det.confidence > 0)
                label += " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";

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

    if (!mask_overlay.empty())
        cv::addWeighted(output, 0.9, mask_overlay, 0.3, 0, output);

    return output;
}

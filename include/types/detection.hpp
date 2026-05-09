#pragma once

#include <map>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <utils/reflect_utils.hpp>

struct Detection
{
    int class_id{-1};
    float confidence{0.f};
    cv::Rect2f bbox{};
    std::string class_name{};
    cv::Mat mask{};

    // MOT specific
    int64_t frame_id{-1};
    int64_t track_id{-1};
    cv::Point3f position{0.0f, 0.0f, 0.0f};

    // Reid specific
    std::vector<float> features{};

    // Multi-label classification
    std::map<int, std::string> labels{};

    // Display
    cv::Size size{}; // set for absolute bbox

    cv::Scalar getClassColor() const
    {
        return getColorById(class_id);
    }

    cv::Scalar getTrackColor() const
    {
        return getColorById(track_id);
    }

    static cv::Scalar getColorById(int id)
    {
        srand(id);
        return cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
    }

    // For MOT file I/O
    friend std::istream &operator>>(std::istream &is, Detection &detection)
    {
        std::string field;

        std::getline(is, field, ',');
        detection.frame_id = std::stoi(field);

        std::getline(is, field, ',');
        detection.track_id = std::stoi(field);

        std::getline(is, field, ',');
        detection.bbox.x = std::stof(field);

        std::getline(is, field, ',');
        detection.bbox.y = std::stof(field);

        std::getline(is, field, ',');
        detection.bbox.width = std::stof(field);

        std::getline(is, field, ',');
        detection.bbox.height = std::stof(field);

        std::getline(is, field, ',');
        detection.confidence = std::stof(field);

        std::getline(is, field, ',');
        detection.position.x = std::stof(field);

        std::getline(is, field, ',');
        detection.position.y = std::stof(field);

        std::getline(is, field);
        detection.position.z = std::stof(field);

        detection.size = cv::Size(detection.bbox.width, detection.bbox.height);

        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const Detection &detection)
    {
        os << detection.frame_id << "," << detection.track_id << ","
           << detection.bbox.x << "," << detection.bbox.y << ","
           << detection.bbox.width << "," << detection.bbox.height << ","
           << detection.confidence << ","
           << detection.position.x << "," << detection.position.y << "," << detection.position.z;
        return os;
    }
};

template <>
struct rfl::Reflector<Detection>
{
    struct ReflType
    {
        int class_id{-1};
        float confidence{0.f};
        cv::Rect2f bbox{};
        std::string class_name{};
        int64_t frame_id{-1};
        int64_t track_id{-1};
        cv::Point3f position{};
        std::vector<float> features{};
        std::map<int, std::string> labels{};
    };

    static ReflType from(const Detection &d) noexcept
    {
        return {d.class_id, d.confidence, d.bbox, d.class_name,
                d.frame_id, d.track_id, d.position, d.features, d.labels};
    }

    static Detection to(const ReflType &r) noexcept
    {
        Detection d;
        d.class_id = r.class_id;
        d.confidence = r.confidence;
        d.bbox = r.bbox;
        d.class_name = r.class_name;
        d.frame_id = r.frame_id;
        d.track_id = r.track_id;
        d.position = r.position;
        d.features = r.features;
        d.labels = r.labels;
        return d;
    }
};

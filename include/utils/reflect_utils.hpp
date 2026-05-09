#pragma once

#include <chrono>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <rfl.hpp>

namespace rfl {

template <>
struct Reflector<cv::Rect2f> {
    struct ReflType {
        float x{0.f}, y{0.f}, width{0.f}, height{0.f};
    };
    static ReflType from(const cv::Rect2f& r) noexcept {
        return {r.x, r.y, r.width, r.height};
    }
    static cv::Rect2f to(const ReflType& r) noexcept {
        return cv::Rect2f(r.x, r.y, r.width, r.height);
    }
};

template <>
struct Reflector<cv::Size> {
    struct ReflType {
        int width{0}, height{0};
    };
    static ReflType from(const cv::Size& s) noexcept {
        return {s.width, s.height};
    }
    static cv::Size to(const ReflType& r) noexcept {
        return cv::Size(r.width, r.height);
    }
};

template <>
struct Reflector<cv::Point3f> {
    struct ReflType {
        float x{0.f}, y{0.f}, z{0.f};
    };
    static ReflType from(const cv::Point3f& p) noexcept {
        return {p.x, p.y, p.z};
    }
    static cv::Point3f to(const ReflType& r) noexcept {
        return cv::Point3f(r.x, r.y, r.z);
    }
};

template <>
struct Reflector<std::chrono::system_clock::time_point> {
    struct ReflType {
        int64_t ms{0};
    };
    static ReflType from(const std::chrono::system_clock::time_point& tp) noexcept {
        return {std::chrono::duration_cast<std::chrono::milliseconds>(
            tp.time_since_epoch()).count()};
    }
    static std::chrono::system_clock::time_point to(const ReflType& r) noexcept {
        return std::chrono::system_clock::time_point{std::chrono::milliseconds{r.ms}};
    }
};

} // namespace rfl

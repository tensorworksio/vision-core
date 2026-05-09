#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

struct ImageTransform
{
public:
    virtual ~ImageTransform() = default;

    // Forward transform
    virtual cv::Mat forward(const cv::Mat &input) const = 0;

    // Reverse transform for different types of input
    virtual cv::Rect2f backward(const cv::Rect2f &input) const = 0;
    virtual cv::Mat backward(const cv::Mat &mask) const = 0;
    virtual cv::Point2f backward(const cv::Point2f &input) const = 0;

    // Clone support for polymorphism
    virtual std::unique_ptr<ImageTransform> clone() const = 0;
};

struct LetterboxTransform : public ImageTransform
{
    cv::Size target_size;
    cv::Scalar padding_color;
    mutable cv::Size input_size;

    mutable float scale;
    mutable int offsetX, offsetY;

    LetterboxTransform(cv::Size size, cv::Scalar color = cv::Scalar(114, 114, 114))
        : target_size(size), padding_color(color) {}

    cv::Mat forward(const cv::Mat &img) const override
    {
        input_size = img.size();
        scale = std::min(
            static_cast<float>(target_size.height) / input_size.height,
            static_cast<float>(target_size.width) / input_size.width);

        cv::Size scaled_size(
            static_cast<int>(std::round(input_size.width * scale)),
            static_cast<int>(std::round(input_size.height * scale)));

        offsetX = (target_size.width - scaled_size.width) / 2;
        offsetY = (target_size.height - scaled_size.height) / 2;

        cv::Mat out(target_size, img.type(), padding_color);
        cv::resize(img,
                   out(cv::Rect(offsetX, offsetY, scaled_size.width, scaled_size.height)),
                   scaled_size,
                   0, 0,
                   cv::INTER_LINEAR);
        return out;
    }

    cv::Point2f backward(const cv::Point2f &point) const override
    {
        return cv::Point2f(
            (point.x - offsetX) / scale,
            (point.y - offsetY) / scale);
    }

    cv::Rect2f backward(const cv::Rect2f &bbox) const override
    {
        return cv::Rect2f(
            (bbox.x - offsetX) / scale,
            (bbox.y - offsetY) / scale,
            bbox.width / scale,
            bbox.height / scale);
    }

    cv::Mat backward(const cv::Mat &mask) const override
    {
        cv::Mat out;

        cv::Size scaled_size(
            static_cast<int>(std::round(input_size.width * scale)),
            static_cast<int>(std::round(input_size.height * scale)));

        // Extract the region without padding
        cv::Mat unpadded = mask(cv::Rect(offsetX, offsetY, scaled_size.width, scaled_size.height));

        // Resize to the original size
        cv::resize(unpadded, out, input_size, 0, 0, cv::INTER_LINEAR);
        return out;
    }

    std::unique_ptr<ImageTransform> clone() const override
    {
        return std::make_unique<LetterboxTransform>(*this);
    }
};
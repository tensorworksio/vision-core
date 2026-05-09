#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "utils/transform_utils.hpp"

class TransformTest : public ::testing::Test
{
protected:
    cv::Mat small_small_image; // 320x480 (both dimensions smaller)
    cv::Mat big_small_image;   // 800x480 (wider, shorter)
    cv::Mat small_big_image;   // 320x800 (narrower, taller)
    cv::Mat big_big_image;     // 800x800 (both dimensions bigger)
    cv::Size target_size{640, 640};
    cv::Scalar padding_color{114, 114, 114};

    void SetUp() override
    {
        // Create test image with both dimensions smaller (320x480)
        small_small_image = cv::Mat(480, 320, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::rectangle(small_small_image, cv::Point(80, 120), cv::Point(240, 360), cv::Scalar(0, 0, 255), 2);

        // Create test image with bigger width, smaller height (800x480)
        big_small_image = cv::Mat(480, 800, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::rectangle(big_small_image, cv::Point(200, 120), cv::Point(600, 360), cv::Scalar(0, 0, 255), 2);

        // Create test image with smaller width, bigger height (320x800)
        small_big_image = cv::Mat(800, 320, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::rectangle(small_big_image, cv::Point(80, 200), cv::Point(240, 600), cv::Scalar(0, 0, 255), 2);

        // Create test image with both dimensions bigger (800x800)
        big_big_image = cv::Mat(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));
        cv::rectangle(big_big_image, cv::Point(200, 200), cv::Point(600, 600), cv::Scalar(0, 0, 255), 2);
    }
};

TEST_F(TransformTest, LetterboxSmallSmallTransform)
{
    LetterboxTransform transform(target_size, padding_color);

    // Apply forward transform
    cv::Mat result = transform.forward(small_small_image);

    // Check output dimensions
    EXPECT_EQ(result.size(), target_size);

    // Check scale calculation
    float expected_scale = std::min(
        static_cast<float>(target_size.width) / small_small_image.cols,   // 640/320 = 2.0
        static_cast<float>(target_size.height) / small_small_image.rows); // 640/480 ≈ 1.33
    EXPECT_FLOAT_EQ(transform.scale, expected_scale);

    // Test backward transformation of a point
    cv::Point2f transformed_point(transform.offsetX + 160 * transform.scale,
                                  transform.offsetY + 240 * transform.scale);
    cv::Point2f original_point = transform.backward(transformed_point);
    EXPECT_NEAR(original_point.x, 160.0f, 0.1f);
    EXPECT_NEAR(original_point.y, 240.0f, 0.1f);

    // Test backward transformation of a rectangle
    cv::Rect2f transformed_rect(
        transform.offsetX + 80 * transform.scale,
        transform.offsetY + 120 * transform.scale,
        160 * transform.scale,
        240 * transform.scale);
    cv::Rect2f original_rect = transform.backward(transformed_rect);
    EXPECT_NEAR(original_rect.x, 80.0f, 0.1f);
    EXPECT_NEAR(original_rect.y, 120.0f, 0.1f);
    EXPECT_NEAR(original_rect.width, 160.0f, 0.1f);
    EXPECT_NEAR(original_rect.height, 240.0f, 0.1f);

    // Test backward transformation of a mask
    cv::Mat mask = cv::Mat::zeros(target_size, CV_8UC1);
    cv::rectangle(mask,
                  cv::Point(transform.offsetX + 80 * transform.scale,
                            transform.offsetY + 120 * transform.scale),
                  cv::Point(transform.offsetX + 240 * transform.scale,
                            transform.offsetY + 360 * transform.scale),
                  cv::Scalar(255), -1);
    cv::Mat original_mask = transform.backward(mask);
    EXPECT_EQ(original_mask.size(), small_small_image.size());
}

TEST_F(TransformTest, LetterboxBigSmallTransform)
{
    LetterboxTransform transform(target_size, padding_color);

    // Apply forward transform
    cv::Mat result = transform.forward(big_small_image);

    // Check output dimensions
    EXPECT_EQ(result.size(), target_size);

    // Check scale calculation
    float expected_scale = std::min(
        static_cast<float>(target_size.width) / big_small_image.cols,   // 640/800 = 0.8
        static_cast<float>(target_size.height) / big_small_image.rows); // 640/480 ≈ 1.33
    EXPECT_FLOAT_EQ(transform.scale, expected_scale);

    // Test backward transformation of a point and rectangle
    cv::Point2f transformed_point(transform.offsetX + 400 * transform.scale,
                                  transform.offsetY + 240 * transform.scale);
    cv::Point2f original_point = transform.backward(transformed_point);
    EXPECT_NEAR(original_point.x, 400.0f, 0.1f);
    EXPECT_NEAR(original_point.y, 240.0f, 0.1f);

    cv::Rect2f transformed_rect(
        transform.offsetX + 200 * transform.scale,
        transform.offsetY + 120 * transform.scale,
        400 * transform.scale,
        240 * transform.scale);
    cv::Rect2f original_rect = transform.backward(transformed_rect);
    EXPECT_NEAR(original_rect.x, 200.0f, 0.1f);
    EXPECT_NEAR(original_rect.y, 120.0f, 0.1f);
    EXPECT_NEAR(original_rect.width, 400.0f, 0.1f);
    EXPECT_NEAR(original_rect.height, 240.0f, 0.1f);

    // Test backward transformation of a mask
    cv::Mat mask = cv::Mat::zeros(target_size, CV_8UC1);
    cv::rectangle(mask,
                  cv::Point(transform.offsetX + 200 * transform.scale,
                            transform.offsetY + 120 * transform.scale),
                  cv::Point(transform.offsetX + 600 * transform.scale,
                            transform.offsetY + 360 * transform.scale),
                  cv::Scalar(255), -1);
    cv::Mat original_mask = transform.backward(mask);
    EXPECT_EQ(original_mask.size(), big_small_image.size());
}

TEST_F(TransformTest, LetterboxSmallBigTransform)
{
    LetterboxTransform transform(target_size, padding_color);

    // Apply forward transform
    cv::Mat result = transform.forward(small_big_image);

    // Check output dimensions
    EXPECT_EQ(result.size(), target_size);

    // Check scale calculation
    float expected_scale = std::min(
        static_cast<float>(target_size.width) / small_big_image.cols,   // 640/320 = 2.0
        static_cast<float>(target_size.height) / small_big_image.rows); // 640/800 = 0.8
    EXPECT_FLOAT_EQ(transform.scale, expected_scale);

    // Test backward transformation of a point and rectangle
    cv::Point2f transformed_point(transform.offsetX + 160 * transform.scale,
                                  transform.offsetY + 400 * transform.scale);
    cv::Point2f original_point = transform.backward(transformed_point);
    EXPECT_NEAR(original_point.x, 160.0f, 0.1f);
    EXPECT_NEAR(original_point.y, 400.0f, 0.1f);

    cv::Rect2f transformed_rect(
        transform.offsetX + 80 * transform.scale,
        transform.offsetY + 200 * transform.scale,
        160 * transform.scale,
        400 * transform.scale);
    cv::Rect2f original_rect = transform.backward(transformed_rect);
    EXPECT_NEAR(original_rect.x, 80.0f, 0.1f);
    EXPECT_NEAR(original_rect.y, 200.0f, 0.1f);
    EXPECT_NEAR(original_rect.width, 160.0f, 0.1f);
    EXPECT_NEAR(original_rect.height, 400.0f, 0.1f);

    // Test backward transformation of a mask
    cv::Mat mask = cv::Mat::zeros(target_size, CV_8UC1);
    cv::rectangle(mask,
                  cv::Point(transform.offsetX + 80 * transform.scale,
                            transform.offsetY + 200 * transform.scale),
                  cv::Point(transform.offsetX + 240 * transform.scale,
                            transform.offsetY + 600 * transform.scale),
                  cv::Scalar(255), -1);
    cv::Mat original_mask = transform.backward(mask);
    EXPECT_EQ(original_mask.size(), small_big_image.size());
}

TEST_F(TransformTest, LetterboxBigBigTransform)
{
    LetterboxTransform transform(target_size, padding_color);

    // Apply forward transform
    cv::Mat result = transform.forward(big_big_image);

    // Check output dimensions
    EXPECT_EQ(result.size(), target_size);

    // Check scale calculation
    float expected_scale = std::min(
        static_cast<float>(target_size.width) / big_big_image.cols,   // 640/800 = 0.8
        static_cast<float>(target_size.height) / big_big_image.rows); // 640/800 = 0.8
    EXPECT_FLOAT_EQ(transform.scale, expected_scale);

    // Test backward transformation of a point and rectangle
    cv::Point2f transformed_point(transform.offsetX + 400 * transform.scale,
                                  transform.offsetY + 400 * transform.scale);
    cv::Point2f original_point = transform.backward(transformed_point);
    EXPECT_NEAR(original_point.x, 400.0f, 0.1f);
    EXPECT_NEAR(original_point.y, 400.0f, 0.1f);

    cv::Rect2f transformed_rect(
        transform.offsetX + 200 * transform.scale,
        transform.offsetY + 200 * transform.scale,
        400 * transform.scale,
        400 * transform.scale);
    cv::Rect2f original_rect = transform.backward(transformed_rect);
    EXPECT_NEAR(original_rect.x, 200.0f, 0.1f);
    EXPECT_NEAR(original_rect.y, 200.0f, 0.1f);
    EXPECT_NEAR(original_rect.width, 400.0f, 0.1f);
    EXPECT_NEAR(original_rect.height, 400.0f, 0.1f);

    // Test backward transformation of a mask
    cv::Mat mask = cv::Mat::zeros(target_size, CV_8UC1);
    cv::rectangle(mask,
                  cv::Point(transform.offsetX + 200 * transform.scale,
                            transform.offsetY + 200 * transform.scale),
                  cv::Point(transform.offsetX + 600 * transform.scale,
                            transform.offsetY + 600 * transform.scale),
                  cv::Scalar(255), -1);
    cv::Mat original_mask = transform.backward(mask);
    EXPECT_EQ(original_mask.size(), big_big_image.size());
}

TEST_F(TransformTest, LetterboxClone)
{
    LetterboxTransform transform(target_size, padding_color);
    transform.forward(small_small_image); // Initialize internal state

    // Clone the transform
    std::unique_ptr<ImageTransform> cloned = transform.clone();
    auto *letterbox_clone = dynamic_cast<LetterboxTransform *>(cloned.get());

    // Verify the clone has the same properties
    ASSERT_NE(letterbox_clone, nullptr);
    EXPECT_EQ(letterbox_clone->target_size, transform.target_size);
    EXPECT_EQ(letterbox_clone->input_size, transform.input_size);
    EXPECT_FLOAT_EQ(letterbox_clone->scale, transform.scale);
    EXPECT_EQ(letterbox_clone->offsetX, transform.offsetX);
    EXPECT_EQ(letterbox_clone->offsetY, transform.offsetY);
    EXPECT_EQ(letterbox_clone->padding_color, transform.padding_color);
}
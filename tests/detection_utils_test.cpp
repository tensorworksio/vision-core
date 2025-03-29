#include <gtest/gtest.h>
#include <utils/detection_utils.hpp>
#include <opencv2/opencv.hpp>

class DetectionUtilsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create sample relative coordinates
        rel_bbox = cv::Rect2f(0.25f, 0.25f, 0.5f, 0.5f); // x, y, width, height

        // Create a sample mask (3x3 matrix)
        rel_mask = (cv::Mat_<float>(3, 3) << 0.1, 0.6, 0.1,
                    0.6, 0.9, 0.6,
                    0.1, 0.6, 0.1);
    }

    cv::Rect2f rel_bbox;
    cv::Mat rel_mask;
};

TEST_F(DetectionUtilsTest, GetAbsoluteBboxTest)
{
    cv::Size size(400, 300);
    cv::Rect abs_bbox = getAbsoluteBbox(rel_bbox, size);

    EXPECT_EQ(abs_bbox.x, 100);      // 0.25 * 400
    EXPECT_EQ(abs_bbox.y, 75);       // 0.25 * 300
    EXPECT_EQ(abs_bbox.width, 200);  // 0.5 * 400
    EXPECT_EQ(abs_bbox.height, 150); // 0.5 * 300
}

TEST_F(DetectionUtilsTest, GetAbsoluteMaskTest)
{
    cv::Size size(4, 4);
    float threshold = 0.5;

    cv::Mat abs_mask = getAbsoluteMask(rel_mask, size, threshold);

    EXPECT_EQ(abs_mask.size(), size);
    EXPECT_EQ(abs_mask.type(), CV_8U);

    // Check that thresholding worked (values should be 0 or 1)
    for (int i = 0; i < abs_mask.rows; i++)
    {
        for (int j = 0; j < abs_mask.cols; j++)
        {
            uchar val = abs_mask.at<uchar>(i, j);
            EXPECT_TRUE(val == 0 || val == 1);
        }
    }
}

TEST_F(DetectionUtilsTest, GetAbsoluteMaskEmptyTest)
{
    cv::Mat empty_mask;
    cv::Size size(4, 4);

    cv::Mat abs_mask = getAbsoluteMask(empty_mask, size);

    EXPECT_TRUE(abs_mask.empty());
}

TEST_F(DetectionUtilsTest, LetterboxBasicTest)
{
    cv::Mat input = cv::Mat::ones(100, 200, CV_8UC3);
    cv::Size new_shape(300, 300);
    cv::Scalar color(114, 114, 114);

    cv::Mat output = letterbox(input, new_shape, color, false, false, true, 32);

    EXPECT_EQ(output.size(), new_shape);
    EXPECT_EQ(output.type(), CV_8UC3);
}

TEST_F(DetectionUtilsTest, LetterboxScaleUpTest)
{
    cv::Mat input = cv::Mat::ones(100, 100, CV_8UC3);
    cv::Size new_shape(200, 200);
    cv::Scalar color(114, 114, 114);

    cv::Mat output = letterbox(input, new_shape, color, false, false, true, 32);

    EXPECT_EQ(output.size(), new_shape);
    EXPECT_EQ(output.type(), CV_8UC3);
}

TEST_F(DetectionUtilsTest, LetterboxNoScaleUpTest)
{
    cv::Mat input = cv::Mat::ones(300, 300, CV_8UC3);
    cv::Size new_shape(200, 200);
    cv::Scalar color(114, 114, 114);
    bool scale_up = false;
    cv::Mat output = letterbox(input, new_shape, color, false, false, scale_up, 32);

    // Should maintain original size since scaleup is false
    EXPECT_EQ(output.size(), new_shape);
    EXPECT_EQ(output.type(), CV_8UC3);
}

TEST_F(DetectionUtilsTest, LetterboxScaleFillTest)
{
    cv::Mat input = cv::Mat::ones(100, 200, CV_8UC3);
    cv::Size new_shape(300, 300);
    cv::Scalar color(114, 114, 114);
    bool scale_fill = true;

    cv::Mat output = letterbox(input, new_shape, color, false, scale_fill, true, 32);

    EXPECT_EQ(output.size(), new_shape);
    EXPECT_EQ(output.type(), CV_8UC3);
}
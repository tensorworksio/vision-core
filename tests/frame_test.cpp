#include <gtest/gtest.h>
#include <types/frame.hpp>
#include <types/detection.hpp>
#include <utils/draw_utils.hpp>
#include <rfl/json.hpp>

class FrameTest : public ::testing::Test
{
protected:
    FrameTest() : test_image(cv::Mat::zeros(100, 100, CV_8UC3)),
                  frame(test_image)
    {
    }

    void SetUp() override
    {
        // Create a sample detection
        Detection det;
        det.bbox = cv::Rect2f(0.25f, 0.25f, 0.5f, 0.5f); // Center box taking up 50% of image
        det.class_id = 1;
        det.class_name = "test_class";
        det.confidence = 0.95f;
        det.track_id = 42;

        // Create a simple mask (white square in the middle)
        det.mask = cv::Mat::zeros(10, 10, CV_32F);
        det.mask(cv::Rect(3, 3, 4, 4)) = 1.0f;

        detections.push_back(det);
    }

    cv::Mat test_image;
    Frame frame;
    std::vector<Detection> detections;
};

TEST_F(FrameTest, ConstructorTest)
{
    EXPECT_EQ(frame.size.width, 100);
    EXPECT_EQ(frame.size.height, 100);
    EXPECT_EQ(frame.width(), 100);
    EXPECT_EQ(frame.height(), 100);
    EXPECT_FALSE(frame.image.empty());
}

TEST_F(FrameTest, TimestampTest)
{
    auto ts = frame.getTimestamp();
    auto ts_ms = frame.getTimestampMs();

    EXPECT_GT(ts_ms, 0);
    EXPECT_EQ(ts_ms, std::chrono::duration_cast<std::chrono::milliseconds>(
                         ts.time_since_epoch())
                         .count());
}

TEST_F(FrameTest, DrawTest)
{
    // Test drawing with class colors
    cv::Mat result = drawDetections(frame, detections, false, true);

    EXPECT_EQ(result.size(), frame.size);
    EXPECT_EQ(result.type(), CV_8UC3);
    EXPECT_FALSE(result.empty());

    // Convert to grayscale before comparing
    cv::Mat gray1, gray2;
    cv::cvtColor(test_image, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(result, gray2, cv::COLOR_BGR2GRAY);

    // Verify that something was actually drawn
    cv::Mat diff;
    cv::compare(gray1, gray2, diff, cv::CMP_NE);
    EXPECT_GT(cv::countNonZero(diff), 0);
}

TEST_F(FrameTest, DrawWithTrackColorsTest)
{
    // Test drawing with track colors
    cv::Mat result = drawDetections(frame, detections, true, true);

    EXPECT_EQ(result.size(), frame.size);
    EXPECT_EQ(result.type(), CV_8UC3);
    EXPECT_FALSE(result.empty());
}

TEST_F(FrameTest, DrawWithoutLabelsTest)
{
    // Test drawing without labels
    cv::Mat result = drawDetections(frame, detections, false, false);

    EXPECT_EQ(result.size(), frame.size);
    EXPECT_EQ(result.type(), CV_8UC3);
    EXPECT_FALSE(result.empty());
}

TEST_F(FrameTest, DrawEmptyDetectionsTest)
{
    std::vector<Detection> empty_detections;
    cv::Mat result = drawDetections(frame, empty_detections);

    EXPECT_EQ(result.size(), frame.size);
    EXPECT_EQ(result.type(), CV_8UC3);

    // Convert to grayscale before comparing
    cv::Mat gray1, gray2;
    cv::cvtColor(test_image, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(result, gray2, cv::COLOR_BGR2GRAY);

    // Images should be identical since nothing was drawn
    cv::Mat diff;
    cv::compare(gray1, gray2, diff, cv::CMP_NE);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST_F(FrameTest, FrameIdTest)
{
    Frame::frame_counter = 0; // Reset counter for this test

    Frame frame1;
    Frame frame2(test_image);

    EXPECT_EQ(frame1.getId(), 0);
    EXPECT_EQ(frame2.getId(), 1);
}

TEST_F(FrameTest, ROIOperatorAbsolute)
{
    Frame frame(test_image);

    // Test valid ROI
    cv::Rect roi(25, 25, 50, 50);
    cv::Mat roi_result = frame(roi);
    EXPECT_EQ(roi_result.cols, roi.width);
    EXPECT_EQ(roi_result.rows, roi.height);

    // Test ROI that exceeds image boundaries
    cv::Rect oversized_roi(75, 75, 50, 50);
    cv::Mat safe_roi = frame(oversized_roi);
    EXPECT_LE(safe_roi.cols + oversized_roi.x, frame.width());
    EXPECT_LE(safe_roi.rows + oversized_roi.y, frame.height());
}

TEST_F(FrameTest, ROIOperatorRelative)
{
    Frame frame(test_image);

    // Test valid relative ROI
    cv::Rect2f rel_roi(0.25f, 0.25f, 0.5f, 0.5f);
    cv::Mat roi_result = frame(rel_roi);

    // Expected size should be half of the original dimensions
    EXPECT_EQ(roi_result.cols, frame.width() / 2);
    EXPECT_EQ(roi_result.rows, frame.height() / 2);

    // Test relative ROI that exceeds image boundaries
    cv::Rect2f oversized_rel_roi(0.8f, 0.8f, 0.3f, 0.3f);
    cv::Mat safe_roi = frame(oversized_rel_roi);
    EXPECT_GT(safe_roi.cols, 0);
    EXPECT_GT(safe_roi.rows, 0);
}

TEST_F(FrameTest, DetectionsDefaultEmpty)
{
    Frame f;
    EXPECT_TRUE(f.detections.empty());
}

TEST_F(FrameTest, DetectionsCanBeAdded)
{
    frame.detections = detections;
    ASSERT_EQ(frame.detections.size(), 1u);
    EXPECT_EQ(frame.detections[0].class_id, 1);
    EXPECT_FLOAT_EQ(frame.detections[0].confidence, 0.95f);
    EXPECT_EQ(frame.detections[0].class_name, "test_class");
}

TEST_F(FrameTest, DetectionSerializationRoundtrip)
{
    const Detection& det = detections[0];
    const auto json_str = rfl::json::write(det);
    EXPECT_FALSE(json_str.empty());

    const auto result = rfl::json::read<Detection>(json_str);
    ASSERT_TRUE(result.has_value());
    const Detection& det2 = result.value();

    EXPECT_EQ(det2.class_id, det.class_id);
    EXPECT_FLOAT_EQ(det2.confidence, det.confidence);
    EXPECT_EQ(det2.class_name, det.class_name);
    EXPECT_EQ(det2.track_id, det.track_id);
    EXPECT_FLOAT_EQ(det2.bbox.x, det.bbox.x);
    EXPECT_FLOAT_EQ(det2.bbox.y, det.bbox.y);
    EXPECT_FLOAT_EQ(det2.bbox.width, det.bbox.width);
    EXPECT_FLOAT_EQ(det2.bbox.height, det.bbox.height);
}

TEST_F(FrameTest, FrameSerializationRoundtrip)
{
    Frame f;
    f.id = 42;
    f.detections = detections;

    const auto json_str = rfl::json::write(f);
    EXPECT_FALSE(json_str.empty());
    EXPECT_NE(json_str.find("detections"), std::string::npos);

    const auto result = rfl::json::read<Frame>(json_str);
    ASSERT_TRUE(result.has_value());
    const Frame& f2 = result.value();

    EXPECT_EQ(f2.id, 42);
    ASSERT_EQ(f2.detections.size(), 1u);
    EXPECT_EQ(f2.detections[0].class_id, 1);
    EXPECT_FLOAT_EQ(f2.detections[0].confidence, 0.95f);
    EXPECT_EQ(f2.detections[0].class_name, "test_class");
    EXPECT_EQ(f2.detections[0].track_id, 42);
}

TEST_F(FrameTest, FrameTimestampSerializationRoundtrip)
{
    Frame f;
    auto now = std::chrono::system_clock::now();
    // Truncate to millisecond precision to match serialization
    now = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    f.timestamp = now;

    const auto json_str = rfl::json::write(f);
    const auto result = rfl::json::read<Frame>(json_str);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result.value().getTimestampMs(), f.getTimestampMs());
}

TEST_F(FrameTest, FrameSizeSerializationRoundtrip)
{
    const auto json_str = rfl::json::write(frame);
    const auto result = rfl::json::read<Frame>(json_str);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result.value().width(), frame.width());
    EXPECT_EQ(result.value().height(), frame.height());
}

TEST_F(FrameTest, DetectionLabelsSerializationRoundtrip)
{
    Detection det;
    det.class_id = 0;
    det.class_name = "multi";
    det.labels = {{0, "cat"}, {1, "animal"}};

    const auto json_str = rfl::json::write(det);
    const auto result = rfl::json::read<Detection>(json_str);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result.value().labels.size(), 2u);
    EXPECT_EQ(result.value().labels.at(0), "cat");
    EXPECT_EQ(result.value().labels.at(1), "animal");
}

TEST_F(FrameTest, DetectionFeaturesSerializationRoundtrip)
{
    Detection det;
    det.features = {0.1f, 0.2f, 0.3f};

    const auto json_str = rfl::json::write(det);
    const auto result = rfl::json::read<Detection>(json_str);
    ASSERT_TRUE(result.has_value());

    ASSERT_EQ(result.value().features.size(), 3u);
    EXPECT_FLOAT_EQ(result.value().features[0], 0.1f);
    EXPECT_FLOAT_EQ(result.value().features[2], 0.3f);
}
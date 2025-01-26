#include <gtest/gtest.h>
#include <utils/geometry_utils.hpp>

class GeometryUtilsTest : public testing::Test
{
protected:
    cv::Rect2f rect1{0, 0, 2, 2}; // Square at origin
    cv::Rect2f rect2{1, 1, 2, 2}; // Overlapping square
    cv::Rect2f rect3{3, 3, 2, 2}; // Non-overlapping square
    cv::Rect2f rect4{0, 0, 0, 0}; // Zero area

    std::vector<float> vec1{1.0f, 0.0f}; // Unit vector along x
    std::vector<float> vec2{0.0f, 1.0f}; // Unit vector along y
    std::vector<float> vec3{1.0f, 1.0f}; // 45-degree vector
    std::vector<float> vec4{0.0f, 0.0f}; // Zero vector
};

TEST_F(GeometryUtilsTest, IoUOverlapping)
{
    float iou = getIoU(rect1, rect2);
    EXPECT_NEAR(iou, 0.14285714f, EPSILON);
}

TEST_F(GeometryUtilsTest, IoUNonOverlapping)
{
    float iou = getIoU(rect1, rect3);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

TEST_F(GeometryUtilsTest, IoUSameRect)
{
    float iou = getIoU(rect1, rect1);
    EXPECT_FLOAT_EQ(iou, 1.0f);
}

TEST_F(GeometryUtilsTest, IoUZeroArea)
{
    float iou = getIoU(rect1, rect4);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

TEST_F(GeometryUtilsTest, CosineSimilarityOrthogonal)
{
    float sim = cosineSimilarity(vec1, vec2);
    EXPECT_FLOAT_EQ(sim, 0.5f);
}

TEST_F(GeometryUtilsTest, CosineSimilaritySame)
{
    float sim = cosineSimilarity(vec1, vec1);
    EXPECT_FLOAT_EQ(sim, 1.0f);
}

TEST_F(GeometryUtilsTest, CosineSimilarity45Degrees)
{
    float sim = cosineSimilarity(vec1, vec3);
    EXPECT_NEAR(sim, 0.853553f, EPSILON);
}

TEST_F(GeometryUtilsTest, CosineSimilarityZeroVector)
{
    float sim = cosineSimilarity(vec1, vec4);
    EXPECT_FLOAT_EQ(sim, 0.0f);
}
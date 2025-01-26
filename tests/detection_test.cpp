#include <gtest/gtest.h>
#include <types/detection.hpp>

TEST(DetectionTest, DefaultConstructor)
{
    Detection det;
    EXPECT_EQ(det.class_id, -1);
    EXPECT_EQ(det.id, -1);
    EXPECT_EQ(det.frame, -1);
    EXPECT_TRUE(det.class_name.empty());
}

TEST(DetectionTest, ColorGeneration)
{
    Detection det1, det2;
    det1.id = 1;
    det1.class_id = 1;

    det2.id = 2;
    det2.class_id = 1;

    EXPECT_EQ(det1.getClassColor(), det2.getClassColor());
    EXPECT_NE(det1.getTrackColor(), det2.getTrackColor());
}

TEST(DetectionTest, StreamOperator)
{
    std::stringstream ss("1,2,10.0,20.0,30.0,40.0,0.9,1.0,2.0,3.0\n");
    Detection det;
    ss >> det;

    EXPECT_EQ(det.frame, 1);
    EXPECT_EQ(det.id, 2);
    EXPECT_FLOAT_EQ(det.bbox.x, 10.0f);
    EXPECT_FLOAT_EQ(det.bbox.y, 20.0f);
    EXPECT_FLOAT_EQ(det.bbox.width, 30.0f);
    EXPECT_FLOAT_EQ(det.bbox.height, 40.0f);
    EXPECT_FLOAT_EQ(det.confidence, 0.9f);
    EXPECT_FLOAT_EQ(det.position.x, 1.0f);
    EXPECT_FLOAT_EQ(det.position.y, 2.0f);
    EXPECT_FLOAT_EQ(det.position.z, 3.0f);
}
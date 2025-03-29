#include <gtest/gtest.h>
#include <utils/vector_utils.hpp>
#include <vector>

class VectorUtilsTest : public testing::Test
{
protected:
    std::vector<float> vec1{1.0f, 2.0f, 3.0f};
    std::vector<float> vec2{4.0f, 5.0f, 6.0f};
    std::vector<float> empty{};
    float scalar = 2.0f;
};

TEST_F(VectorUtilsTest, VectorAddition)
{
    auto result = vector_ops::add(vec1, vec2);
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 5.0f);
    EXPECT_FLOAT_EQ(result[1], 7.0f);
    EXPECT_FLOAT_EQ(result[2], 9.0f);
    EXPECT_THROW(vector_ops::add(vec1, empty), std::invalid_argument);
}

TEST_F(VectorUtilsTest, ScalarAddition)
{
    auto result = vector_ops::add(vec1, scalar);
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 3.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f);
    EXPECT_FLOAT_EQ(result[2], 5.0f);
}

TEST_F(VectorUtilsTest, VectorMultiplication)
{
    auto result = vector_ops::mul(vec1, vec2);
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 4.0f);
    EXPECT_FLOAT_EQ(result[1], 10.0f);
    EXPECT_FLOAT_EQ(result[2], 18.0f);
    EXPECT_THROW(vector_ops::add(vec1, empty), std::invalid_argument);
}

TEST_F(VectorUtilsTest, ScalarMultiplication)
{
    auto result = vector_ops::mul(vec1, scalar);
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 4.0f);
    EXPECT_FLOAT_EQ(result[2], 6.0f);
}

TEST_F(VectorUtilsTest, DotProduct)
{
    float result = vector_ops::dot(vec1, vec2);
    EXPECT_FLOAT_EQ(result, 32.0f); // (1*4 + 2*5 + 3*6)
    EXPECT_THROW(vector_ops::dot(vec1, empty), std::invalid_argument);
}

TEST_F(VectorUtilsTest, Normalize)
{
    auto result = vector_ops::normalize(vec1); // sqrt(14)
    EXPECT_FLOAT_EQ(result[0], vec1[0] / std::sqrt(14.f));
    EXPECT_FLOAT_EQ(result[1], vec1[1] / std::sqrt(14.f));
    EXPECT_FLOAT_EQ(result[2], vec1[2] / std::sqrt(14.f));
}

TEST_F(VectorUtilsTest, Compose)
{
    float alpha = 0.3f;
    auto result = vector_ops::compose(vec1, vec2, alpha);
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 1.0f * 0.3f + 4.0f * 0.7f);
    EXPECT_THROW(vector_ops::compose(vec1, empty, alpha), std::invalid_argument);
}

TEST_F(VectorUtilsTest, Sum)
{
    float result = vector_ops::sum(vec1);
    EXPECT_FLOAT_EQ(result, 6.0f); // 1 + 2 + 3
    EXPECT_FLOAT_EQ(vector_ops::sum(empty), 0.0f);
}

TEST_F(VectorUtilsTest, Mean)
{
    float result = vector_ops::mean(vec1);
    EXPECT_FLOAT_EQ(result, 2.0f); // (1 + 2 + 3) / 3
    EXPECT_FLOAT_EQ(vector_ops::mean(empty), 0.0f);
}

TEST_F(VectorUtilsTest, Max)
{
    float result = vector_ops::max(vec1);
    EXPECT_FLOAT_EQ(result, 3.0f);
}

TEST_F(VectorUtilsTest, Exp)
{
    auto result = vector_ops::exp(vec1);
    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], std::exp(1.0f));
    EXPECT_FLOAT_EQ(result[1], std::exp(2.0f));
    EXPECT_FLOAT_EQ(result[2], std::exp(3.0f));
}

TEST_F(VectorUtilsTest, Slice)
{
    auto result = vector_ops::slice(vec1, 0, 2);
    EXPECT_EQ(result.size(), 2);
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
}

TEST_F(VectorUtilsTest, SigmoidEmptyVector)
{
    auto result = vector_ops::sigmoid(empty);
    EXPECT_TRUE(result.empty());
}

TEST_F(VectorUtilsTest, SigmoidSingleElement)
{
    std::vector<float> input{0.0f};
    auto result = vector_ops::sigmoid(input);
    EXPECT_EQ(result.size(), 1);
    EXPECT_NEAR(result[0], 0.5f, 1e-6);
}

TEST_F(VectorUtilsTest, SigmoidMultipleElements)
{
    std::vector<float> input{-2.0f, 0.0f, 2.0f};
    auto result = vector_ops::sigmoid(input);
    EXPECT_EQ(result.size(), 3);
    EXPECT_NEAR(result[0], 0.119203f, 1e-6);
    EXPECT_NEAR(result[1], 0.5f, 1e-6);
    EXPECT_NEAR(result[2], 0.880797f, 1e-6);
}

TEST_F(VectorUtilsTest, SoftmaxEmptyVector)
{
    auto result = vector_ops::softmax(empty);
    EXPECT_TRUE(result.empty());
}

TEST_F(VectorUtilsTest, SoftmaxSingleElement)
{
    std::vector<float> input{1.0f};
    auto result = vector_ops::softmax(input);
    EXPECT_EQ(result.size(), 1);
    EXPECT_NEAR(result[0], 1.0f, 1e-6);
}

TEST_F(VectorUtilsTest, SoftmaxMultipleElements)
{
    std::vector<float> input{1.0f, 2.0f, 3.0f};
    auto result = vector_ops::softmax(input);
    EXPECT_EQ(result.size(), 3);
    EXPECT_NEAR(result[0], 0.090031f, 1e-6);
    EXPECT_NEAR(result[1], 0.244728f, 1e-6);
    EXPECT_NEAR(result[2], 0.665241f, 1e-6);
}

TEST_F(VectorUtilsTest, SoftmaxLargeNumbers)
{
    std::vector<float> input{100.0f, 100.0f, 100.0f};
    auto result = vector_ops::softmax(input);
    EXPECT_EQ(result.size(), 3);
    for (const auto &val : result)
    {
        EXPECT_NEAR(val, 1.0f / 3.0f, 1e-6);
    }
}

TEST_F(VectorUtilsTest, SoftmaxNegativeNumbers)
{
    std::vector<float> input{-1.0f, -2.0f, -3.0f};
    auto result = vector_ops::softmax(input);
    EXPECT_EQ(result.size(), 3);
    EXPECT_NEAR(result[0], 0.665241f, 1e-6);
    EXPECT_NEAR(result[1], 0.244728f, 1e-6);
    EXPECT_NEAR(result[2], 0.090031f, 1e-6);
}

TEST_F(VectorUtilsTest, MaxEmptyVector)
{
    EXPECT_THROW(vector_ops::max(empty), std::invalid_argument);
}
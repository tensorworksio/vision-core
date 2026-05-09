#pragma once

#include <cmath>
#include <vector>
#include <ranges>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace vector_ops
{

    // Element-wise addition of two vectors
    template <typename T>
    inline std::vector<T> add(const std::vector<T> &a, const std::vector<T> &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Vectors must be the same size");
        }

        std::vector<T> result(a.size());
        std::ranges::transform(a, b, result.begin(), std::plus<>{});
        return result;
    }

    // Scalar add
    template <typename T>
    inline std::vector<T> add(const std::vector<T> &vec, T scalar)
    {
        std::vector<T> result(vec.size());
        std::ranges::transform(vec, result.begin(), [scalar](T x)
                       { return x + scalar; });
        return result;
    }

    // Element-wise multiplication of two vectors
    template <typename T>
    inline std::vector<T> mul(const std::vector<T> &a, const std::vector<T> &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Vectors must be the same size");
        }

        std::vector<T> result(a.size());
        std::ranges::transform(a, b, result.begin(), std::multiplies<>{});
        return result;
    }

    // Scalar multiplication
    template <typename T>
    inline std::vector<T> mul(const std::vector<T> &vec, T scalar)
    {
        std::vector<T> result(vec.size());
        std::ranges::transform(vec, result.begin(), [scalar](T x)
                       { return x * scalar; });
        return result;
    }

    // Dot product of two vectors
    template <typename T>
    inline T dot(const std::vector<T> &a, const std::vector<T> &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Vectors must be the same size");
        }

        return std::ranges::fold_left(
            std::views::zip_transform(std::multiplies<>{}, a, b),
            T(0), std::plus<>{});
    }

    // Normalize vector
    template <typename T>
    inline std::vector<T> normalize(const std::vector<T> &vec)
    {
        std::vector<T> result(vec.size());
        T norm = std::sqrt(dot(vec, vec));
        result = mul(vec, T(1) / norm);
        return result;
    }

    // Compose 2 vectors with a weighted average
    template <typename T>
    inline std::vector<T> compose(const std::vector<T> &a, const std::vector<T> &b, T alpha)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("Vectors must be the same size");
        }
        const T alpha_complement = T(1) - alpha;
        std::vector<T> result(a.size());
        std::ranges::transform(a, b, result.begin(), [alpha, alpha_complement](T x, T y)
                       { return alpha * x + alpha_complement * y; });
        return result;
    }

    // Sum vector
    template <typename T>
    inline T sum(const std::vector<T> &vec)
    {
        return std::ranges::fold_left(vec, T(0), std::plus<>{});
    }

    // Mean vector
    template <typename T>
    inline T mean(const std::vector<T> &vec)
    {
        if (vec.empty())
        {
            return T(0);
        }
        T size = static_cast<T>(vec.size());
        return sum(vec) / size;
    }

    // Max vector
    template <typename T>
    inline T max(const std::vector<T> &vec)
    {
        if (vec.empty())
        {
            throw std::invalid_argument("Cannot find maximum of empty vector");
        }
        return std::ranges::max(vec);
    }

    template <typename T>
    inline size_t argmax(const std::vector<T> &vec)
    {
        if (vec.empty())
        {
            throw std::invalid_argument("Cannot find maximum of empty vector");
        }
        return static_cast<size_t>(std::ranges::max_element(vec) - vec.begin());
    }

    // Exp vector
    template <typename T>
    inline std::vector<T> exp(const std::vector<T> &vec)
    {
        std::vector<T> result(vec.size());
        std::ranges::transform(vec, result.begin(), [](T x)
                       { return std::exp(x); });
        return result;
    }

    // Slice vector
    template <typename T>
    inline std::vector<T> slice(const std::vector<T> &vec, int start, int end)
    {
        auto first = vec.begin() + start;
        auto last = vec.begin() + end;
        std::vector<T> sliced(first, last);
        return sliced;
    }

    // Sigmoid
    template <typename T>
    inline std::vector<T> sigmoid(const std::vector<T> &logits)
    {
        if (logits.empty())
        {
            return std::vector<T>();
        }

        std::vector<T> results(logits.size());
        std::ranges::transform(logits, results.begin(),
                       [](T x)
                       {
                           return T(1) / (T(1) + std::exp(-x));
                       });
        return results;
    }

    // Softmax
    template <typename T>
    inline std::vector<T> softmax(const std::vector<T> &logits)
    {
        if (logits.empty())
        {
            return std::vector<T>();
        }

        if (logits.size() == 1)
        {
            return std::vector<T>{T(1)};
        }

        T max_logit = vector_ops::max(logits);
        std::vector<T> exp_values = vector_ops::exp(vector_ops::add(logits, -max_logit));
        T sum_exp = vector_ops::sum(exp_values);

        return vector_ops::mul(exp_values, T(1) / sum_exp);
    }

} // namespace vector_ops

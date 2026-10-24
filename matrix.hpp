//
// Created by cartercpp on 10/24/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <memory_resource>
#include <algorithm>
#include <concepts>
#include <cstddef>
#include "math_vector.hpp"

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class matrix
{
public:

    // CONSTRUCTORS

    explicit matrix(
        std::size_t rows,
        std::size_t columns,
        ValueType value = 0,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource},
        m_rows{rows},
        m_columns{columns}
    {
        m_data.resize(rows * columns, value);
    }

    matrix(
        std::initializer_list<math_vector<ValueType>> data,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(data.size() > 0)
        : m_data{resource},
        m_rows{data.size()},
        m_columns{data.begin()->size()}
    {
        m_data.reserve(m_rows * m_columns);

        for (const auto& rowVec : data)
            for (const ValueType value : rowVec)
                m_data.push_back(value);
    }

    explicit matrix(
        const matrix& other,
        std::pmr::memory_resource* resource
    )
        : m_data{resource},
        m_rows{other.rows()},
        m_columns{other.columns()}
    {
        m_data.reserve(m_rows * m_columns);

        for (std::size_t row = 0; row < m_rows; ++row)
            for (std::size_t column = 0; column < m_columns; ++column)
                m_data.emplace_back(other[row, column]);
    }

    matrix() noexcept
        : m_rows{0}, m_columns{0}
    {}

    matrix(const matrix&) = default;
    matrix& operator=(const matrix&) = default;
    matrix(matrix&&) noexcept = default;
    matrix& operator=(matrix&&) = default;

    // METHODS

    [[nodiscard]] std::size_t rows() const noexcept
    {
        return m_rows;
    }

    [[nodiscard]] std::size_t columns() const noexcept
    {
        return m_columns;
    }

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept
    {
        return m_data.get_allocator().resource();
    }

    [[nodiscard]] ValueType& operator[](const std::size_t row, const std::size_t column)
        pre(row < m_rows)
        pre(column < m_columns)
    {
        return m_data[row * m_columns + column];
    }

    [[nodiscard]] const ValueType& operator[](const std::size_t row, const std::size_t column) const
        pre(row < m_rows)
        pre(column < m_columns)
    {
        return m_data[row * m_columns + column];
    }

    matrix& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef *= scalar;});
        return *this;
    }

    matrix& operator/=(ValueType scalar)
        pre(scalar != 0)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef /= scalar;});
        return *this;
    }

    matrix& operator+=(const matrix& other)
        pre(m_rows == other.rows())
        pre(m_columns == other.columns())
    {
        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] += other.m_data[i];

        return *this;
    }

    matrix& operator-=(const matrix& other)
        pre(m_rows == other.rows())
        pre(m_columns == other.columns())
    {
        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] -= other.m_data[i];

        return *this;
    }

    matrix squared() const
    {
        matrix output(m_rows, m_columns, 0, resource());

        for (std::size_t i = 0; i < m_data.size(); ++i)
            output.m_data[i] = m_data[i] * m_data[i];

        return output;
    }

private:

    std::pmr::vector<ValueType> m_data;
    std::size_t m_rows,
                m_columns;
};

template <typename ValueType>
bool operator==(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    if ((lArg.rows() != rArg.rows()) || (lArg.columns() != rArg.columns()))
        return false;

    for (std::size_t row = 0; row < lArg.rows(); ++row)
        for (std::size_t column = 0; column < lArg.columns(); ++column)
            if (lArg[row, column] != rArg[row, column])
                return false;

    return true;
}

template <typename ValueType>
bool operator!=(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& mat, ValueType scalar)
{
    matrix<ValueType> output{mat, mat.resource()};
    output *= scalar;
    return output;
}

template <typename ValueType>
auto operator*(ValueType scalar, const matrix<ValueType>& mat)
{
    matrix<ValueType> output{mat, mat.resource()};
    output *= scalar;
    return output;
}

template <typename ValueType>
auto operator/(const matrix<ValueType>& mat, ValueType scalar)
{
    matrix<ValueType> output{mat, mat.resource()};
    output /= scalar;
    return output;
}

template <typename ValueType>
auto operator+(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    matrix<ValueType> output{lArg, lArg.resource()};
    output += rArg;
    return output;
}

template <typename ValueType>
auto operator-(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
{
    matrix<ValueType> output{lArg, lArg.resource()};
    output -= rArg;
    return output;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& lArg, const matrix<ValueType>& rArg)
    pre(lArg.columns() == rArg.rows())
    post(output : (output.rows() == lArg.rows()) && (output.columns() == rArg.columns()))
{
    matrix<ValueType> output(lArg.rows(), rArg.columns(), 0, lArg.resource());

    for (std::size_t row = 0; row < lArg.rows(); ++row)
        for (std::size_t column = 0; column < rArg.columns(); ++column)
        {
            ValueType value = 0;

            for (std::size_t i = 0; i < lArg.columns(); ++i)
                value += lArg[row, i] * rArg[i, column];

            output[row, column] = value;
        }

    return output;
}

template <typename ValueType>
auto outer_product(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
    post(output : (output.rows() == lArg.size()) && (output.columns() == rArg.size()))
{
    matrix<ValueType> output(lArg.size(), rArg.size(), 0, lArg.resource());

    for (std::size_t i = 0; i < lArg.size(); ++i)
        for (std::size_t i2 = 0; i2 < rArg.size(); ++i2)
            output[i, i2] = lArg[i] * rArg[i2];

    return output;
}

template <typename ValueType>
auto operator*(const matrix<ValueType>& mat, const math_vector<ValueType>& vec)
    pre(mat.columns() == vec.size())
    post(output : output.size() == mat.rows())
{
    math_vector<ValueType> output(mat.rows(), 0, mat.resource());

    for (std::size_t row = 0; row < mat.rows(); ++row)
    {
        ValueType value = 0;

        for (std::size_t i = 0; i < vec.size(); ++i)
            value += mat[row, i] * vec[i];

        output[row] = value;
    }

    return output;
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& vec, const matrix<ValueType>& mat)
    pre(vec.size() == mat.rows())
    post(output : output.size() == mat.columns())
{
    math_vector<ValueType> output(mat.columns(), 0, mat.resource());

    for (std::size_t column = 0; column < mat.columns(); ++column)
    {
        ValueType value = 0;

        for (std::size_t i = 0; i < vec.size(); ++i)
            value += mat[i, column] * vec[i];

        output[column] = value;
    }

    return output;
}
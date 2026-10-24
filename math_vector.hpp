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

template <typename ValueType> requires (std::integral<ValueType> || std::floating_point<ValueType>)
class math_vector
{
public:

    // CONSTRUCTORS

    explicit math_vector(
        std::size_t size,
        ValueType value = 0,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource}
    {
        m_data.resize(size, value);
    }

    math_vector(
        std::initializer_list<ValueType> data,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        : m_data{resource}
    {
        m_data.reserve(data.size());
        for (const ValueType value : data)
            m_data.push_back(value);
    }

    explicit math_vector(
        const math_vector& other,
        std::pmr::memory_resource* resource
    )
        : m_data{resource}
    {
        m_data.resize(other.size(), 0);
        for (std::size_t i = 0; i < other.size(); ++i)
            m_data[i] = other[i];
    }

    math_vector() noexcept = default;
    math_vector(const math_vector&) = default;
    math_vector& operator=(const math_vector&) = default;
    math_vector(math_vector&&) noexcept = default;
    math_vector& operator=(math_vector&&) = default;

    // METHODS

    [[nodiscard]] std::size_t size() const noexcept
    {
        return m_data.size();
    }

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept
    {
        return m_data.get_allocator().resource();
    }

    [[nodiscard]] ValueType& operator[](std::size_t index)
        pre(index < m_data.size())
    {
        return m_data[index];
    }

    [[nodiscard]] const ValueType& operator[](std::size_t index) const
        pre(index < m_data.size())
    {
        return m_data[index];
    }

    math_vector& operator*=(ValueType scalar)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef *= scalar;});
        return *this;
    }

    math_vector& operator/=(ValueType scalar)
        pre(scalar != 0)
    {
        std::for_each(m_data.begin(), m_data.end(), [scalar](ValueType& valueRef){valueRef /= scalar;});
        return *this;
    }

    math_vector& operator+=(const math_vector& other)
        pre(m_data.size() == other.size())
    {
        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] += other[i];

        return *this;
    }

    math_vector& operator-=(const math_vector& other)
        pre(m_data.size() == other.size())
    {
        for (std::size_t i = 0; i < m_data.size(); ++i)
            m_data[i] -= other[i];

        return *this;
    }

    math_vector multiply(const math_vector& other) const
        pre(m_data.size() == other.size())
    {
        math_vector output(m_data.size(), 0, resource());

        for (std::size_t i = 0; i < m_data.size(); ++i)
            output[i] = m_data[i] * other[i];

        return output;
    }

    auto begin() noexcept
    {
        return m_data.begin();
    }

    auto end() noexcept
    {
        return m_data.end();
    }

    auto begin() const noexcept
    {
        return m_data.begin();
    }

    auto end() const noexcept
    {
        return m_data.end();
    }

private:

    std::pmr::vector<ValueType> m_data;
};

template <typename ValueType>
bool operator==(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    if (lArg.size() != rArg.size())
        return false;

    for (std::size_t i = 0; i < lArg.size(); ++i)
        if (lArg[i] != rArg[i])
            return false;

    return true;
}

template <typename ValueType>
bool operator!=(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
{
    return !(lArg == rArg);
}

template <typename ValueType>
ValueType operator*(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
    pre(lArg.size() == rArg.size())
{
    ValueType output = 0;

    for (std::size_t i = 0; i < lArg.size(); ++i)
        output += lArg[i] * rArg[i];

    return output;
}

template <typename ValueType>
auto operator*(const math_vector<ValueType>& vec, ValueType scalar)
{
    math_vector<ValueType> output{vec, vec.resource()};
    output *= scalar;
    return output;
}

template <typename ValueType>
auto operator*(ValueType scalar, const math_vector<ValueType>& vec)
{
    math_vector<ValueType> output{vec, vec.resource()};
    output *= scalar;
    return output;
}

template <typename ValueType>
auto operator/(const math_vector<ValueType>& vec, ValueType scalar)
    pre(scalar != 0)
{
    math_vector<ValueType> output{vec, vec.resource()};
    output /= scalar;
    return output;
}

template <typename ValueType>
auto operator+(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
    pre(lArg.size() == rArg.size())
{
    math_vector<ValueType> output{lArg, lArg.resource()};
    output += rArg;
    return output;
}

template <typename ValueType>
auto operator-(const math_vector<ValueType>& lArg, const math_vector<ValueType>& rArg)
    pre(lArg.size() == rArg.size())
{
    math_vector<ValueType> output{lArg, lArg.resource()};
    output -= rArg;
    return output;
}
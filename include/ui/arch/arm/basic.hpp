#ifndef AMT_UI_ARCH_ARM_BASIC_HPP
#define AMT_UI_ARCH_ARM_BASIC_HPP

#include "ui/base.hpp"
#include <cstddef>
#include <cstdint>

namespace ui::arm::internal {

    template <typename T, typename U>
    struct WideningResult {
        using type = T;
    };

    template <>
    struct WideningResult<std::byte, std::byte> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::byte> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::byte, std::uint16_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint8_t, std::uint8_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::uint8_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::uint8_t, std::uint16_t> {
        using type = std::uint16_t;
    };

    template <>
    struct WideningResult<std::int8_t, std::int8_t> {
        using type = std::int16_t;
    };

    template <>
    struct WideningResult<std::int16_t, std::int8_t> {
        using type = std::int16_t;
    };

    template <>
    struct WideningResult<std::int8_t, std::int16_t> {
        using type = std::int16_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::uint16_t> {
        using type = std::uint32_t;
    };

    template <>
    struct WideningResult<std::uint32_t, std::uint16_t> {
        using type = std::uint32_t;
    };

    template <>
    struct WideningResult<std::uint16_t, std::uint32_t> {
        using type = std::uint32_t;
    };

    template <>
    struct WideningResult<std::int16_t, std::int16_t> {
        using type = std::int32_t;
    };

    template <>
    struct WideningResult<std::int32_t, std::int16_t> {
        using type = std::int32_t;
    };

    template <>
    struct WideningResult<std::int16_t, std::int32_t> {
        using type = std::int32_t;
    };

    template <>
    struct WideningResult<std::uint32_t, std::uint32_t> {
        using type = std::uint64_t;
    };

    template <>
    struct WideningResult<std::uint64_t, std::uint32_t> {
        using type = std::uint64_t;
    };

    template <>
    struct WideningResult<std::uint32_t, std::uint64_t> {
        using type = std::uint64_t;
    };

    template <>
    struct WideningResult<std::int32_t, std::int32_t> {
        using type = std::int64_t;
    };

    template <>
    struct WideningResult<std::int64_t, std::int32_t> {
        using type = std::int64_t;
    };

    template <>
    struct WideningResult<std::int32_t, std::int64_t> {
        using type = std::int64_t;
    };

    template <>
    struct WideningResult<std::int64_t, std::int64_t> {
        using type = float;
    };

    template <>
    struct WideningResult<float, float> {
        using type = double;
    };

    template <typename T, typename U = T>
    using widening_result_t = typename WideningResult<T, U>::type;

    template <typename T>
    struct NarrowingResult;

    template <>
    struct NarrowingResult<std::uint16_t> {
        using type = std::uint8_t;
    };

    template <>
    struct NarrowingResult<std::int16_t> {
        using type = std::int8_t;
    };

    template <>
    struct NarrowingResult<std::uint32_t> {
        using type = std::uint16_t;
    };

    template <>
    struct NarrowingResult<std::int32_t> {
        using type = std::int16_t;
    };

    template <>
    struct NarrowingResult<std::uint64_t> {
        using type = std::uint32_t;
    };

    template <>
    struct NarrowingResult<std::int64_t> {
        using type = std::int32_t;
    };

    template <typename T>
    using narrowing_result_t = typename NarrowingResult<T>::type;


    template <bool Round, typename Acc, typename From>
    UI_ALWAYS_INLINE constexpr auto halving_round_helper(From lhs, From rhs, auto&& op) noexcept -> From {
        auto l = static_cast<Acc>(lhs);
        auto r = static_cast<Acc>(rhs);
        auto sum = op(l, r) + Round;
        return static_cast<From>(sum >> 1);
    }
} // namespace ui::arm::internal

#endif // AMT_UI_ARCH_ARM_BASIC_HPP

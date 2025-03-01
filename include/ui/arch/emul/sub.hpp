#ifndef AMT_ARCH_EMUL_SUB_HPP
#define AMT_ARCH_EMUL_SUB_HPP

#include "cast.hpp"
#include "ui/arch/basic.hpp"
#include "ui/base.hpp"
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace ui::emul {
// MARK: Wrapping Subtraction
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            return static_cast<T>(l - r);
        }, lhs, rhs);
    }
// !MARK

// MARK: Widening Subtraction
    template <std::size_t N, std::integral T, std::integral U>
        requires ((((sizeof(T) * 2) == sizeof(U)) || ((sizeof(U) * 2) == sizeof(T)) || sizeof(T) == sizeof(U)) && (std::is_signed_v<T> == std::is_signed_v<U>))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
        using result_t = internal::widening_result_t<T, U>;
        return map([](auto l, auto r) -> result_t {
            return static_cast<result_t>(l) - static_cast<result_t>(r);
        }, lhs, rhs);
    }
// !MARK

// MARK: Narrowing Subtraction
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto halving_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) -> T {
            return static_cast<T>(internal::halving_round_helper(l, r, op::sub_t{}));
        }, lhs, rhs);
    }

    /**
     *  @returns upper half bits of the vector register
    */
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto high_narrowing_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept {
        using result_t = internal::narrowing_result_t<T>; 
        return map([](auto l, auto r) -> result_t {
            return static_cast<result_t>((l - r) >> (sizeof(result_t) * 8));
        }, lhs, rhs);
    }
// !MARK

// MARK: Saturating Subtraction
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            auto res = static_cast<T>(l - r);
            static constexpr auto bits = sizeof(T) * 8 - 1;
            static constexpr auto sign_bit = T(1) << bits;
            if constexpr (std::is_signed_v<T>) {
                if (((res ^ l) & sign_bit) && ((l ^ r) & sign_bit)) {
                    return static_cast<T>((l >> bits) ^ ~sign_bit);
                }
                return res;
            } else {
                return res > l ? T(0) : res;
            }
        }, lhs, rhs);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_SUB_HPP

#ifndef AMT_ARCH_EMUL_SUB_HPP
#define AMT_ARCH_EMUL_SUB_HPP

#include "cast.hpp"
#include "../basic.hpp"
#include "../../base.hpp"
#include "../../features.hpp"
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace ui::emul {
// MARK: Wrapping Subtraction
    template <std::size_t N, typename T>
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

// MARK: Subtraction with carry
    template <std::integral T>
        requires std::is_unsigned_v<T>
    UI_ALWAYS_INLINE auto subc(
        T const& a,
        T const& b,
        T carry = {}
    ) noexcept -> std::pair<T /*result*/, T /*carry*/> {
        if constexpr (sizeof(T) == 1) {
            auto l = static_cast<std::uint16_t>(a);
            auto r = static_cast<std::uint16_t>(b) + static_cast<std::uint16_t>(carry);
            auto s = l - r;
            return { static_cast<T>(s), l < r };
        } else if constexpr (sizeof(T) == 2) {
            auto l = static_cast<std::uint32_t>(a);
            auto r = static_cast<std::uint32_t>(b) + static_cast<std::uint32_t>(carry);
            auto s = l - r;
            return { static_cast<T>(s), l < r };
        } else {
            #ifdef UI_ARCH_64BIT
            if constexpr (sizeof(T) == 4) {
                auto l = static_cast<std::uint64_t>(a);
                auto r = static_cast<std::uint64_t>(b) + static_cast<std::uint64_t>(carry);
                auto s = l - r;
                return { static_cast<T>(s), l < r };
            #ifdef UI_HAS_INT128
            } else if constexpr (sizeof(T) == 8) {
                static constexpr auto bits = (sizeof(T) * CHAR_BIT);
                auto l = uint128_t(a);
                auto r = uint128_t(b) + uint128_t(carry);
                auto s = l - r;
                return { static_cast<T>(s), l < r };
            #endif
            }
            #endif
            auto sum = a - carry;
            auto c0 = a < carry;
            auto c1 = sum < b;
            sum = sum - b;
            return { sum, c0 | c1 };
        }
    }

    template <std::size_t N, std::integral T>
        requires std::is_unsigned_v<T>
    UI_ALWAYS_INLINE auto subc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T carry = {}
    ) noexcept -> std::pair<Vec<N, T>, T /*carry*/> {
        if constexpr (N == 1) {
            auto [sum, c] = adcc(a.val, b.val, carry);
            return { Vec<N, T>(sum), c };
        } else {
            auto [l, lc] = subc(a.lo, b.lo, carry);
            auto [h, hc] = subc(a.hi, b.hi, lc);
            return { join(l, h), hc };
        }
    }
// !Mark
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_SUB_HPP

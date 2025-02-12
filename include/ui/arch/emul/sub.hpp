#ifndef AMT_ARCH_EMUL_SUB_HPP
#define AMT_ARCH_EMUL_SUB_HPP

#include "cast.hpp"
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
        requires (((sizeof(T) << 1) == sizeof(U)) || ((sizeof(U) << 1) == sizeof(T)))
    UI_ALWAYS_INLINE auto widening_sub(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
        using result_t = internal::widening_result_t<T, U>;
        return map([](auto l, auto r) {
            return static_cast<result_t>(l) - static_cast<result_t>(r);
        }, lhs, rhs);
    }
// !MARK

// MARK: Narrowing Subtraction
    template <bool Round = false, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto halving_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using acc_t = internal::widening_result_t<T>;
        return map([](auto l, auto r) {
            return internal::halving_round_helper<Round, acc_t>(l, r, op::sub_t{});
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
        return map([](auto l, auto r) {
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
            using type = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
            auto diff = static_cast<type>(l) - static_cast<type>(r);
            static constexpr auto min = static_cast<type>(std::numeric_limits<T>::min());
            static constexpr auto max = static_cast<type>(std::numeric_limits<T>::max());
            return static_cast<T>(
                std::clamp<type>(diff, min, max)
            );
        }, lhs, rhs);
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_SUB_HPP

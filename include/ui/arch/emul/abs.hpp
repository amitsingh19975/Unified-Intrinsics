#ifndef AMT_ARCH_EMUL_ABS_HPP
#define AMT_ARCH_EMUL_ABS_HPP

#include "cast.hpp"
#include <concepts>
#include <limits>
#include <type_traits>

namespace ui::emul {

    namespace internal {
        using namespace ::ui::internal;

        template <typename T>
        UI_ALWAYS_INLINE static constexpr auto abs_diff_helper(
            T l,
            T r
        ) -> T {
            if constexpr (std::integral<T>) {
                using utype = std::make_unsigned_t<T>;
                auto l0 = static_cast<utype>(l);
                auto r0 = static_cast<utype>(r);
                auto lr = l0 - r0;
                auto rl = r0 - l0;
                auto tmp = l >= r ? lr : rl;
                return static_cast<T>(tmp);
            } else {
                return l >= r ? l - r : r - l;
            }
        }
    } // namespace internal

// MARK: Difference
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](T l, T r) -> T {
            return internal::abs_diff_helper(l, r);
        }, lhs, rhs);
    }
// !MARK

// MARK: Widening Difference
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto widening_abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return map([](auto l, auto r) {
            auto l0 = static_cast<result_t>(l);
            auto r0 = static_cast<result_t>(r);
            return static_cast<result_t>(
                l > r ? (l0 - r0) : (r0 - l0)
            );
        }, lhs, rhs);
    }
// !MARK

// MARK: Absolute difference and Accumulate
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](T a, T l, T r){
            auto res = internal::abs_diff_helper(l, r);
            return static_cast<T>(a + res);
        }, acc, lhs, rhs);
    }
// !MARK

// MARK: Absolute Value
    template <std::size_t N, typename T>
        requires ((std::integral<T> && std::is_signed_v<T>) || std::floating_point<T>)
    UI_ALWAYS_INLINE static constexpr auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using std::abs;
        using ui::abs;
        return map([](auto v_) {
            return static_cast<T>(abs(v_));
        }, v);
    }
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return v;
    }

    template <std::size_t N, std::integral T>
        requires std::is_signed_v<T>
    UI_ALWAYS_INLINE static constexpr auto sat_abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](T v_) -> T {
            static constexpr T sign_bit = static_cast<T>(T(1) << (sizeof(T) * 8 - 1));
            if (v_ == sign_bit) return std::numeric_limits<T>::max();
            return std::abs(v_);
        }, v);
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto sat_abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return v;
    }
// !MARK
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_ABS_HPP

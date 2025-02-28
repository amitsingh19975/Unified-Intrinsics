#ifndef AMT_UI_ARCH_EMUL_ADD_HPP
#define AMT_UI_ARCH_EMUL_ADD_HPP

#include "cast.hpp"
#include "ui/arch/basic.hpp"
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace ui::emul {
    
    namespace internal {
        using namespace ::ui::internal;
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) { return static_cast<T>(l + r); }, lhs, rhs);
    }

// MARK: Widening Addition
    template <std::size_t N, std::integral T, std::integral U>
    UI_ALWAYS_INLINE static constexpr auto widening_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T, U>> {
        using result_t = internal::widening_result_t<T, U>;
        return map([](auto l, auto r) { 
            return static_cast<result_t>(static_cast<result_t>(l) + static_cast<result_t>(r));
        }, lhs, rhs);
    }
// !MAKR

// MARK: Halving Widening Addition
    template <bool Round = false, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using acc_t = internal::widening_result_t<T>;
        return map([](auto l, auto r) {
            return internal::halving_round_helper<Round, acc_t>(l, r, op::add_t{});
        }, lhs, rhs); 
    }
// !MAKR

// MARK: High-bit Narrowing Addition
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto high_narrowing_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::narrowing_result_t<T>> {
        using result_t = internal::narrowing_result_t<T>;
        return map([](auto l, auto r) {
            return (static_cast<result_t>((l + r) >> (sizeof(result_t) * 8)));
        }, lhs, rhs); 
    }
// !MAKR

// MARK: Saturation Addition
    namespace internal {
        template <std::integral T>
        UI_ALWAYS_INLINE static constexpr auto sat_add_helper(
            T l,
            T r
        ) noexcept -> T {
            auto sum = static_cast<T>(l + r);
            static constexpr auto bits = sizeof(T) * 8 - 1;
            static constexpr auto min = std::numeric_limits<T>::min();
            static constexpr auto max = std::numeric_limits<T>::max();
            if constexpr (std::is_signed_v<T>) {
                auto mask = ((l ^ sum) & ~(l ^ r)) >> bits;
                auto sat = (l >> bits) ^ std::numeric_limits<T>::max();
                return static_cast<T>((sum & ~mask) | (sat & mask));
            } else {
                return sum < l ? std::numeric_limits<T>::max() : sum;
            }
        }

    } // namespace internal
    template <std::size_t N, std::integral T, std::integral U>
    UI_ALWAYS_INLINE static constexpr auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {
                return internal::sat_add_helper(l, r);
            } else if constexpr (std::is_signed_v<T>) {
                // T: signed, U: unsigned
                static constexpr auto min = std::numeric_limits<T>::max();
                if (r >= static_cast<U>(min)) return min;
                auto tr = static_cast<T>(r);
                return internal::sat_add_helper(l, tr); 
            } else {
                // T: unsigned, U: signed
                static constexpr auto min = std::numeric_limits<U>::max();
                if (l >= static_cast<T>(min)) return min;
                auto tr = static_cast<T>(r);
                return internal::sat_add_helper(l, tr); 
            }
        }, lhs, rhs);
    }
// !MAKR

// MARK: Pairwise Addition
    template <std::size_t N, typename T>
        requires (N != 1)
    UI_ALWAYS_INLINE static constexpr auto padd(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            return { static_cast<T>(lhs[0] + lhs[1]), static_cast<T>(rhs[0] + rhs[1]) };
        } else {
            return join(
                padd(lhs.lo, lhs.hi),
                padd(rhs.lo, rhs.hi)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto fold(
        Vec<N, T> const& v,
        op::padd_t op
    ) noexcept -> T {
        if constexpr (N == 1) return v.val;
        else return fold(v.lo, op) + fold(v.hi, op);
    }


// !MAKR

// MARK: Pairwise Addition
    template <std::size_t N, std::integral T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto widening_padd(
        Vec<N, T> const& v
    ) noexcept -> Vec<N / 2, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 2) {
            return {
                .val = static_cast<result_t>(static_cast<result_t>(v[0]) + static_cast<result_t>(v[1]))
            };
        } else {
            return join(
                widening_padd(v.lo),
                widening_padd(v.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE static constexpr auto widening_padd(
        Vec<    N, internal::widening_result_t<T>> const& a,
        Vec<2 * N, T> v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return {
                .val = static_cast<result_t>(a.val + static_cast<result_t>(v.lo.val) + static_cast<result_t>(v.hi.val))
            };
        } else {
            return join(
                widening_padd(a.lo, v.lo),
                widening_padd(a.hi, v.hi)
            );
        }
    }
// !MAKR

// MARK: Addition across vector
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::add_t op
    ) noexcept -> T {
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& v_
        ) -> T {
            return static_cast<T>((v_[Is] +...));
        };
        return helper(std::make_index_sequence<N>{}, v);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto widening_fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::add_t op
    ) noexcept -> internal::widening_result_t<T> {
        using result_t = internal::widening_result_t<T>;
        constexpr auto helper = []<std::size_t... Is>(
            std::index_sequence<Is...>,
            Vec<N, T> const& v_
        ) -> result_t {
            return static_cast<result_t>((static_cast<result_t>(v_[Is]) +...));
        };
        return helper(std::make_index_sequence<N>{}, v);
    }
// !MAKR
} // namespace ui::emul

#endif // AMT_UI_ARCH_EMUL_ADD_HPP

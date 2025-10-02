#ifndef AMT_UI_ARCH_EMUL_ADD_HPP
#define AMT_UI_ARCH_EMUL_ADD_HPP

#include "cast.hpp"
#include <climits>
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
    template <std::size_t N, std::integral T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE static constexpr auto halving_add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using acc_t = internal::widening_result_t<T>;
        return map([](auto l, auto r) -> acc_t {
            return static_cast<acc_t>(internal::halving_round_helper(l, r, op::add_t{}));
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
        return map([](auto l, auto r) -> result_t {
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
        requires (sizeof(T) == sizeof(U))
    UI_ALWAYS_INLINE static constexpr auto sat_add(
        Vec<N, T> const& lhs,
        Vec<N, U> const& rhs
    ) noexcept -> Vec<N, T> {
        return map([](auto l, auto r) {
            if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {
                return internal::sat_add_helper(l, r);
            } else if constexpr (std::is_signed_v<T>) {
                // T: signed, U: unsigned
                static constexpr auto max = std::numeric_limits<T>::max();
                if (r >= static_cast<U>(max)) return max;
                auto tr = static_cast<T>(r);
                return static_cast<T>(internal::sat_add_helper(l, tr));
            } else {
                // T: unsigned, U: signed
                #ifndef UI_HAS_INT128
                if constexpr (sizeof(T) == 8) {
                    static constexpr auto max = static_cast<T>(std::numeric_limits<U>::max());
                    if (l > max) return std::numeric_limits<T>::max();
                    if (r < 0) {
                        if (static_cast<U>(l) >= r) return T(0);
                        return static_cast<T>(l + r);
                    }
                    auto tl = static_cast<U>(l);
                    return static_cast<T>(internal::sat_add_helper(tl, r));
                } else
                #endif
                {

                    #ifdef UI_HAS_INT128
                    using type = std::conditional_t<sizeof(T) == 8, ui::int128_t, std::int64_t>;
                    #else
                    using type = std::conditional_t<std::is_signed_v<T>, std::int64_t, std::uint64_t>;
                    #endif
                    auto a = static_cast<type>(l);
                    auto b = static_cast<type>(r);
                    return static_cast<T>(std::clamp<type>(
                        a + b,
                        type(0),
                        static_cast<type>(std::numeric_limits<T>::max())
                    ));
                }
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
        Vec<2 * N, T> const& v
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

// MARK: Addition with carry
    template <std::size_t N, std::integral T>
        requires (std::is_unsigned_v<class Tp>)
    UI_ALWAYS_INLINE auto addc(
        T a,
        T b,
        T carry = {}
    ) noexcept -> std::pair<T /*result*/, T /*carry*/> {
        if constexpr (sizeof(T) == 1) {
            auto l = static_cast<std::uint16_t>(a);
            auto r = static_cast<std::uint16_t>(b) + static_cast<std::uint16_t>(carry);
            auto s = l + r;
            static constexpr auto bits = (sizeof(T) * CHAR_BIT);
            return { static_cast<T>(s), static_cast<T>(s >> bits) };
        } else if constexpr (sizeof(T) == 2) {
            auto l = static_cast<std::uint32_t>(a);
            auto r = static_cast<std::uint32_t>(b) + static_cast<std::uint32_t>(carry);
            auto s = l + r;
            static constexpr auto bits = (sizeof(T) * CHAR_BIT);
            return { static_cast<T>(s), static_cast<T>(s >> bits) };
        } else {
            #ifdef UI_ARCH_64BIT
            if constexpr (sizeof(T) == 4) {
                auto l = static_cast<std::uint64_t>(a);
                auto r = static_cast<std::uint64_t>(b) + static_cast<std::uint64_t>(carry);
                auto s = l + r;
                static constexpr auto bits = (sizeof(T) * CHAR_BIT);
                return { static_cast<T>(s), static_cast<T>(s >> bits) };
            }
            #endif
            auto sum = a + b;
            auto c0 = sum < a;
            sum = sum + carry;
            auto c1 = sum < carry;
            return { sum, sum < a };
        }
    }

    template <std::size_t N, std::integral T>
        requires (std::is_unsigned_v<class Tp>)
    UI_ALWAYS_INLINE auto addc(
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

#endif // AMT_UI_ARCH_EMUL_ADD_HPP

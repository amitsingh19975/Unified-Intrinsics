#ifndef AMT_UI_ARCH_WASM_ABS_HPP
#define AMT_UI_ARCH_WASM_ABS_HPP

#include "cast.hpp"
#include "../emul/abs.hpp"
#include "cmp.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::abs(v);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto m = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_abs(m)); 
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_abs(m)); 
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(abs(
                        cast<float>(v)
                    ));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_abs(m));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_abs(m));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_i32x4_abs(m));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_i64x2_abs(m));
                    }
                } else {
                    return v;
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return abs(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            return join(
                abs<false>(v.lo),
                abs<false>(v.hi)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto bitwise_select(
        mask_t<N, T> const& cond,
        Vec<N, T> const& true_,
        Vec<N, T> const& false_
    ) noexcept -> Vec<N, T>;

    template <bool Merge, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <bool Merge, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <bool Merge, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto add(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_xor(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (std::is_signed_v<T>) {
            auto s0 = sub(lhs, rhs);
            auto s1 = sub(rhs, lhs);
            auto c = cmp(lhs, rhs, op::greater_t{});
            auto temp = bitwise_select(c, s0, s1);
            return temp;
        } else {
            auto s0 = sat_sub(lhs, rhs);
            auto s1 = sat_sub(rhs, lhs);
            return bitwise_or(s0, s1);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto widening_abs_diff(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return abs_diff(cast<result_t>(lhs), cast<result_t>(rhs));
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        auto res = abs_diff(lhs, rhs);
        return add(acc, res);
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto abs_acc_diff(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        auto res = widening_abs_diff(lhs, rhs);
        return add(acc, res);
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_abs(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        auto m = Vec<N, T>::load(static_cast<T>(1) << (sizeof(T) * 8 - 1));
        auto a = abs(v);
        auto c = cmp(a, m, op::equal_t{});
        return bitwise_xor(a, rcast<T>(c));
    }
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_ABS_HPP

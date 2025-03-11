#ifndef AMT_UI_ARCH_WASM_DIV_HPP
#define AMT_UI_ARCH_WASM_DIV_HPP

#include "cast.hpp"
#include "../emul/div.hpp"
#include "rounding.hpp"
#include "mul.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <bool Merge = true, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto div(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(num);
        if constexpr (N == 1) {
            return emul::div(num, den);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_div(to_vec(num), to_vec(den)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_div(to_vec(num), to_vec(den)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(div(cast<float>(num), cast<float>(den)));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return div(from_vec<T>(fit_to_vec(num)), from_vec<T>(fit_to_vec(den))).lo;
            }

            return join(
                div<false>(num.lo, den.lo),
                div<false>(num.hi, den.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto div(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::div(num, den);
        } else {
            if constexpr (sizeof(T) == 8) {
                auto tn = cast<double>(num);
                auto td = cast<double>(den);
                auto res = round<std::float_round_style::round_toward_zero>(div(tn, td));
                return cast<T>(res);
            } else {
                auto tn = cast<float>(num);
                auto td = cast<float>(den);
                auto res = round<std::float_round_style::round_toward_zero>(div(tn, td));
                return cast<T>(res);
            }
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto rem(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        auto q = round<std::float_round_style::round_toward_zero>(div(num, den));
        return fused_mul_acc(num, q, den, op::sub_t{});
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto rem(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        return mul_acc(num, div(num, den), den, op::sub_t{});
    }
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_DIV_HPP

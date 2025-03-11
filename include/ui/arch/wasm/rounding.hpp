#ifndef AMT_UI_ARCH_WASM_ROUNDING_HPP
#define AMT_UI_ARCH_WASM_ROUNDING_HPP

#include "cast.hpp"
#include "../emul/rounding.hpp"

namespace ui::wasm {
    template <std::float_round_style mode = std::float_round_style::round_to_nearest, bool Merge = true, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto round(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::round<mode>(v);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto m = to_vec(v);
                if constexpr (std::same_as<T, float>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(wasm_f32x4_trunc(m));
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(wasm_f32x4_nearest(m));
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(wasm_f32x4_ceil(m));
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(wasm_f32x4_floor(m));
                    } else {
                        return from_vec<T>(wasm_f32x4_trunc(m));
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(wasm_f64x2_trunc(m));
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(wasm_f64x2_nearest(m));
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(wasm_f64x2_ceil(m));
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(wasm_f64x2_floor(m));
                    } else {
                        return round<std::numeric_limits<T>::round_style>(v);
                    }
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(round<mode>(cast<float>(v)));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return round<mode>(from_vec<T>(fit_to_vec(v))).lo;
            }

            return join(
                round<mode, false>(v.lo),
                round<mode, false>(v.hi)
            );
        }
    }
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_ROUNDING_HPP

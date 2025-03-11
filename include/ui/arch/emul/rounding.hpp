#ifndef AMT_ARCH_EMUL_ROUNDING_HPP
#define AMT_ARCH_EMUL_ROUNDING_HPP

#include "cast.hpp"
#include <limits>

#ifndef UI_USE_CSTDLIB
    #if !defined(UI_EMPSCRIPTEN)
        #define UI_USE_CSTDLIB
    #endif
#endif

#if defined(UI_USE_CSTDLIB)
#include <cmath>
#include <cfenv>
#endif

namespace ui::emul {
    template <std::float_round_style mode = std::float_round_style::round_to_nearest, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto round(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        #if defined(UI_USE_CSTDLIB)
        auto const old = std::fegetround(); 
        std::fesetround(::ui::internal::convert_rounding_style(mode));
        auto temp = map([](auto v_) -> T {
            if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                return T(std::nearbyint(float(v_))); 
            } else {
                return std::nearbyint(v_); 
            }
        }, v);
        std::fesetround(old);
        return temp;
        #else
            if constexpr (::ui::internal::is_fp16<T>) {
                return cast<T>(round<mode>(cast<float>(v)));
            } else {
                if constexpr (mode == std::float_round_style::round_indeterminate) {
                    return round<std::numeric_limits<T>::round_style>(v);
                } else {
                    return map([](auto v_) -> T {
                        if constexpr (mode == std::float_round_style::round_toward_zero) {
                            if constexpr (std::same_as<T, float>) {
                                return __builtin_truncf(v_);
                            } else {
                                return __builtin_trunc(v_);
                            }
                        } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                            if constexpr (std::same_as<T, float>) {
                                return __builtin_nearbyintf(v_);
                            } else {
                                return __builtin_nearbyint(v_);
                            }
                        } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                            if constexpr (std::same_as<T, float>) {
                                return __builtin_ceilf(v_);
                            } else {
                                return __builtin_ceil(v_);
                            }
                        } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                            if constexpr (std::same_as<T, float>) {
                                return __builtin_floorf(v_);
                            } else {
                                return __builtin_floor(v_);
                            }
                        }
                    }, v);
                }
            }
        #endif
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_ROUNDING_HPP

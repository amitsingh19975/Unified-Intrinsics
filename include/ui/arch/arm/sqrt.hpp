#ifndef AMT_UI_ARCH_ARM_SQRT_HPP
#define AMT_UI_ARCH_ARM_SQRT_HPP

#include "cast.hpp"
#include <cassert>
#include <cfenv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>

namespace ui::arm::neon {

    template<std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sqrt(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            if constexpr (std::same_as<T, float16>) {
                return {
                    .val = T(std::sqrt(float(v.val)))
                };
            } else {
                return {
                    .val = static_cast<T>(std::sqrt(v.val))
                };
            }
        } else {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsqrt_f32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vsqrtq_f32(to_vec(v)));
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vsqrtq_f64(to_vec(v)));
                    }
                } else if constexpr (std::same_as<T, float16>) {
                    if constexpr (N == 4) {
                        return from_vec(vsqrt_f16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec(vsqrtq_f16(to_vec(v)));
                    }
                }
            #endif

            return join(
                sqrt(v.lo),
                sqrt(v.hi)
            );
        }
    }

}

#endif // AMT_UI_ARCH_ARM_SQRT_HPP

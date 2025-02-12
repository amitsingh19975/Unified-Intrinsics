#ifndef AMT_UI_ARCH_ARM_SQRT_HPP
#define AMT_UI_ARCH_ARM_SQRT_HPP

#include "cast.hpp"
#include "../emul/sqrt.hpp"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>

namespace ui::arm::neon {

    template<std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sqrt(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::sqrt(v);
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
                    #ifdef UI_HAS_FLOAT_16
                    if constexpr (N == 4) {
                        return from_vec(vsqrt_f16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return from_vec(vsqrtq_f16(to_vec(v)));
                    }
                    #else
                    return cast<T>(sqrt(cast<float>(v)));
                    #endif
                } else if constexpr (std::same_as<T, bfloat16>) {
                    return cast<T>(sqrt(cast<float>(v)));
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

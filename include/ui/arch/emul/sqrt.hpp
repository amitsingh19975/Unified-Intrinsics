#ifndef AMT_ARCH_EMUL_SQRT_HPP
#define AMT_ARCH_EMUL_SQRT_HPP

#include "cast.hpp"
#include "../../maths.hpp"

#if defined(UI_USE_CSTDLIB)
#include <cmath>
#endif

namespace ui::emul {
    template<std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto sqrt(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                return T(std::sqrt(float(v_)));
            } else {
                if constexpr (std::integral<T>) {
                    return static_cast<T>(maths::isqrt(v_));
                } else {
                #if defined(UI_USE_CSTDLIB)
                    return static_cast<T>(std::sqrt(v_));
                #else
                    if constexpr (std::same_as<T, float>) {
                        return static_cast<T>(__builtin_sqrtf(v_));
                    } else {
                        return static_cast<T>(__builtin_sqrt(v_));
                    }
                #endif
                }
            }
        }, v);
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_SQRT_HPP

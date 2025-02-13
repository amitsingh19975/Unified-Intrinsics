#ifndef AMT_ARCH_EMUL_SQRT_HPP
#define AMT_ARCH_EMUL_SQRT_HPP

#include "cast.hpp"
#include <concepts>

namespace ui::emul {
    template<std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto sqrt(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return map([](auto v_) {
            if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                return T(std::sqrt(float(v_)));
            } else {
                return std::sqrt(v_);
            }
        }, v);
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_SQRT_HPP

#ifndef AMT_ARCH_EMUL_ROUNDING_HPP
#define AMT_ARCH_EMUL_ROUNDING_HPP

#include "cast.hpp"
#include <cmath>
#include <concepts>
#include <cfenv>

namespace ui::emul {
    template <std::float_round_style mode = std::float_round_style::round_to_nearest, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto round(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        auto const old = std::fegetround(); 
        std::fesetround(::ui::internal::convert_rounding_style(mode));
        auto temp = map([](auto v_) {
            return std::nearbyint(v_); 
        }, v);
        std::fesetround(old);
        return temp;
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_ROUNDING_HPP

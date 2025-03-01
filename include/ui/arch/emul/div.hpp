#ifndef AMT_ARCH_EMUL_DIV_HPP
#define AMT_ARCH_EMUL_DIV_HPP

#include "cast.hpp"
#include <concepts>

namespace ui::emul {

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto div(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        return map([](auto n, auto d) -> T {
            if constexpr (std::integral<T>) {
                // floating-point divisions are faster in most of the cases. (Need testing)
                return static_cast<T>(float(n) / float(d));
            } else {
                return n / d;
            }
        }, num, den);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto rem(
        Vec<N, T> const& num,
        Vec<N, T> const& den
    ) noexcept -> Vec<N, T> {
        return map([](auto n, auto d) -> T {
            if constexpr (std::integral<T>) {
                // floating-point divisions are faster in most of the cases. (Need testing)
                return n - static_cast<T>(float(n) / float(d)) * d;
            } else {
                if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    auto n0 = float(n);
                    auto d0 = float(d);
                    return std::fmod(n0, d0);
                } else {
                    return std::fmod(n, d);
                }
            }
        }, num, den);
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_DIV_HPP

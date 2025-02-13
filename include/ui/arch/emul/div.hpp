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
        return map([](auto n, auto d) {
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
        return map([](auto n, auto d) {
            if constexpr (std::integral<T>) {
                // floating-point divisions are faster in most of the cases. (Need testing)
                return n - static_cast<T>(float(n) / float(d));
            } else {
                using type = std::conditional_t<
                    sizeof(T) == 2,
                    std::int16_t,
                    std::conditional_t<
                        sizeof(T) == 4,
                        std::int32_t,
                        std::int64_t
                    >
                >;
                return n - static_cast<T>(static_cast<type>(n / d));
            }
        }, num, den);
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_DIV_HPP

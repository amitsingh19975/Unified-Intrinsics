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

} // namespace ui::emul

#endif // AMT_ARCH_EMUL_DIV_HPP

#ifndef AMT_ARCH_EMUL_LOAD_HPP
#define AMT_ARCH_EMUL_LOAD_HPP

#include "cast.hpp"

namespace ui::emul {

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(T val) noexcept -> Vec<N, T> {
        auto res = Vec<N, T>{};
        std::copy_n(res.data(), N, val);
        return res;
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        auto res = Vec<N, T>{};
        std::copy_n(res.data(), N, v[Lane]);
        return res;
    }

} // namespace ui::emul

#endif // AMT_ARCH_EMUL_LOAD_HPP

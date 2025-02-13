#ifndef AMT_ARCH_EMUL_LOAD_HPP
#define AMT_ARCH_EMUL_LOAD_HPP

#include "cast.hpp"
#include <algorithm>
#include <utility>

namespace ui::emul {

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(T val) noexcept -> Vec<N, T> {
        auto res = Vec<N, T>{};
        std::fill_n(res.data(), N, val);
        return res;
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        auto res = Vec<N, T>{};
        std::fill_n(res.data(), N, v[Lane]);
        return res;
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b
    ) noexcept {
        auto const helper = [&]<std::size_t... Is>(std::index_sequence<Is...>){
            ((
                a[Is] = data[2 * Is + 0],
                b[Is] = data[2 * Is + 1]
            ),...);
        };
        helper(std::make_index_sequence<N>{});
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c
    ) noexcept {
        auto const helper = [&]<std::size_t... Is>(std::index_sequence<Is...>){
            ((
                a[Is] = data[3 * Is + 0],
                b[Is] = data[3 * Is + 1],
                c[Is] = data[3 * Is + 2]
            ),...);
        };
        helper(std::make_index_sequence<N>{});
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c,
        Vec<N, T>& UI_RESTRICT d
    ) noexcept {
        auto const helper = [&]<std::size_t... Is>(std::index_sequence<Is...>){
            ((
                a[Is] = data[4 * Is + 0],
                b[Is] = data[4 * Is + 1],
                c[Is] = data[4 * Is + 2],
                d[Is] = data[4 * Is + 3]
            ),...);
        };
        helper(std::make_index_sequence<N>{});
    }
} // namespace ui::emul

#endif // AMT_ARCH_EMUL_LOAD_HPP

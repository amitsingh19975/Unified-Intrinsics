#ifndef AMT_UI_ARCH_WASM_JOIN_HPP
#define AMT_UI_ARCH_WASM_JOIN_HPP

#include "../../vec_headers.hpp"
#include "../../forward.hpp"
#include <bit>
#include <cstddef>
#include <cstdint>

namespace ui::wasm {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto join_impl(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<2 * N, T> {
        using ret_t = Vec<2 * N, T>;
        static constexpr auto size = sizeof(x);
        constexpr auto cast = [](auto v) -> ret_t {
            return std::bit_cast<ret_t>(v);
        };

        if constexpr (N == 1) {
            return { x, y };
        } else {
            if constexpr (size * 2 == sizeof(v128_t)) {
                auto low = std::bit_cast<std::int64_t>(x);
                auto hi = std::bit_cast<std::int64_t>(y);
                return cast(wasm_i64x2_make(low, hi));
            }
            return { x, y };
        }
    }
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_JOIN_HPP

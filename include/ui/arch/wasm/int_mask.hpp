#ifndef AMT_ARCH_WASM_INT_MASK_HPP
#define AMT_ARCH_WASM_INT_MASK_HPP

#include "cast.hpp"
#include "shift.hpp"

namespace ui {
    template <std::size_t N, typename T>
    inline constexpr IntMask<N, T>::IntMask(mask_t<N, T> const& m) noexcept {
        using namespace emul;
        using mtype = mask_inner_t<T>;

        if constexpr (sizeof(T) == 1) {
            if constexpr (N == 16) {
                mask = static_cast<base_type>(wasm_i8x16_bitmask(to_vec(m)));
                return;
            }
        } else if constexpr (sizeof(T) == 2) {
            if constexpr (N == 8) {
                mask = static_cast<base_type>(wasm_i16x8_bitmask(to_vec(m)));
            }
        } else if constexpr (sizeof(T) == 4) {
            if constexpr (N == 4) {
                mask = static_cast<base_type>(wasm_i32x4_bitmask(to_vec(m)));
            }
        } else if constexpr (sizeof(T) == 8) {
            if constexpr (N == 2) {
                mask = static_cast<base_type>(wasm_i64x2_bitmask(to_vec(m)));
            }
        }

        auto ext = rcast<mtype>(shift_right<7>(rcast<std::make_signed_t<T>>(m)));
        auto helper = [&ext]<std::size_t... Is>(std::index_sequence<Is...>) -> base_type {
            auto res = base_type{};
            ((res |= (base_type(ext[Is] & 1) << Is)),...);
            return res;
        };
        mask = helper(std::make_index_sequence<N>{});
    }
} // namespace ui
#endif // AMT_ARCH_WASM_INT_MASK_HPP

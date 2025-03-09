#ifndef AMT_UI_ARCH_WASM_LOAD_HPP
#define AMT_UI_ARCH_WASM_LOAD_HPP

#include "cast.hpp"
#include "../emul/load.hpp"
#include <type_traits>

namespace ui::wasm {
    template <std::size_t N, typename T, bool Merge = true>
    UI_ALWAYS_INLINE auto load(T val) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(T) * N;
        if constexpr (N == 1) {
            return emul::load<N>(val);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    if constexpr (N == 4) {
                        return from_vec<T>(wasm_f32x4_splat(val));
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (N == 2) {
                        return from_vec<T>(wasm_f64x2_splat(val));
                    }
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return rcast<T>(load<N>(std::bit_cast<std::uint16_t>(val)));
                } else {
                    using utype = std::make_unsigned_t<T>;
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_u8x16_splat(static_cast<utype>(val)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_u16x8_splat(static_cast<utype>(val)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_u32x4_splat(static_cast<utype>(val)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_u64x2_splat(static_cast<utype>(val)));
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t)) {
                return load<2 * N, T>(val).lo;
            }

            auto t = load<N / 2, T, false>(val);
            return join(t, t);
        }
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto load(
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        return load<N>(v[Lane]);
    }
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_LOAD_HPP

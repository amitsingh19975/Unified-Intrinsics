#ifndef AMT_UI_ARCH_WASM_BASIC_HPP
#define AMT_UI_ARCH_WASM_BASIC_HPP

#include "../../vec_headers.hpp"
#include <cstddef>
#include <cstdint>

namespace ui::wasm {
    namespace constants {
        alignas(16) static constexpr std::int8_t mask8_16_even_odd[32] = {
            0,  2,  4,  6,  8,  10, 12, 14,
            1,  3,  5,  7,  9,  11, 13, 15,
            0,  2,  4,  6,  8,  10, 12, 14,
            1,  3,  5,  7,  9,  11, 13, 15,
        };

        alignas(16) static constexpr std::int8_t mask8_32_even_odd[32] = { 
            0,  1,  4,  5,  8,  9,  12, 13,
            2,  3,  6,  7,  10, 11, 14, 15,
            0,  1,  4,  5,  8,  9,  12, 13,
            2,  3,  6,  7,  10, 11, 14, 15,
        };

        static constexpr auto swap_hi_low32 = 0b01'00'11'10; // (2 | (3 << 2) | (0 << 4) | (1 << 6));
    }

    namespace internal {
        UI_ALWAYS_INLINE static auto unpacklo_i8(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i8x16_shuffle(
                a, b,
                0, 16, 1, 17, 2, 18, 3, 19,
                4, 20, 5, 21, 6, 22, 7, 23
            );
        }

        UI_ALWAYS_INLINE static auto unpacklo_i16(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i16x8_shuffle(
                a, b,
                0, 8, 1, 9, 2, 10, 3, 11
            );
        }

        UI_ALWAYS_INLINE static auto unpacklo_i32(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i32x4_shuffle(a, b, 0, 4, 1, 5);
        }

        UI_ALWAYS_INLINE static auto unpacklo_i64(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i64x2_shuffle(a, b, 0, 2);
        }

        UI_ALWAYS_INLINE static auto unpackhi_i8(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i8x16_shuffle(
                a, b,
                8, 24, 9, 25, 10, 26, 11, 27,
                12, 28, 13, 29, 14, 30, 15, 31
            );
        }

        UI_ALWAYS_INLINE static auto unpackhi_i16(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i16x8_shuffle(
                a, b,
                4, 12, 5, 13, 6, 14, 7, 15
            );
        }

        UI_ALWAYS_INLINE static auto unpackhi_i32(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i32x4_shuffle(a, b, 2, 6, 3, 7);
        }

        UI_ALWAYS_INLINE static auto unpackhi_i64(
            v128_t a,
            v128_t b
        ) noexcept -> v128_t {
             return wasm_i64x2_shuffle(a, b, 1, 3);
        }
    } // namespace internal
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_BASIC_HPP

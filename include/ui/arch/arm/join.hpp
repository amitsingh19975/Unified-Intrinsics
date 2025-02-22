#ifndef AMT_UI_ARCH_ARM_JOIN_HPP
#define AMT_UI_ARCH_ARM_JOIN_HPP

#include "../../forward.hpp"
#include "../../vec_headers.hpp"
#include "../../float.hpp"
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::arm::neon { 
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto join_impl(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<2 * N, T> {
        using ret_t = Vec<2 * N, T>;

        if constexpr (N == 1) {
            return { x, y };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(vcombine_f32(std::bit_cast<float32x2_t>(x), std::bit_cast<float32x2_t>(y)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcombine_f16(std::bit_cast<float16x4_t>(x), std::bit_cast<float16x4_t>(y)));
                }
                #else
                return std::bit_cast<ret_t>(join_impl(
                    std::bit_cast<Vec<N, std::uint16_t>>(x),
                    std::bit_cast<Vec<N, std::uint16_t>>(y)
                    )
                );
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                #ifdef UI_HAS_BFLOAT_16
                if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(vcombine_bf16(std::bit_cast<bfloat16x4_t>(x), std::bit_cast<bfloat16x4_t>(y)));
                }
                #else
                return std::bit_cast<ret_t>(join_impl(
                    std::bit_cast<Vec<N, std::uint16_t>>(x),
                    std::bit_cast<Vec<N, std::uint16_t>>(y)
                    )
                );
                #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcombine_s8(std::bit_cast<int8x8_t>(x), std::bit_cast<int8x8_t>(y)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcombine_s16(std::bit_cast<int16x4_t>(x), std::bit_cast<int16x4_t>(y)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcombine_s32(std::bit_cast<int32x2_t>(x), std::bit_cast<int32x2_t>(y)));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(vcombine_u8(std::bit_cast<uint8x8_t>(x), std::bit_cast<uint8x8_t>(y)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(vcombine_u16(std::bit_cast<uint16x4_t>(x), std::bit_cast<uint16x4_t>(y)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(vcombine_u32(std::bit_cast<uint32x2_t>(x), std::bit_cast<uint32x2_t>(y)));
                    }
                }
            }
            return {x, y};
        }
    }
} // namespace ui::arm::neon;

#endif // AMT_UI_ARCH_ARM_JOIN_HPP

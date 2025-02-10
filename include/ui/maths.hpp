#ifndef AMT_UI_MATHS_HPP
#define AMT_UI_MATHS_HPP

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <array>
#include "base.hpp"

#if defined(UI_COMPILER_GCC) || defined(UI_COMPILER_CLANG)
    #if __has_builtin(__builtin_bswap16)
        #define UI_BYTE_SWAP_INTRINSIC_2(VAL) __builtin_bswap16(VAL)
    #else
        #define UI_BYTE_SWAP_INTRINSIC_2(VAL) __builtin_bswap32((VAL) << 16)
    #endif
    #define UI_BYTE_SWAP_INTRINSIC_4(VAL) __builtin_bswap32(VAL)
    #define UI_BYTE_SWAP_INTRINSIC_8(VAL) __builtin_bswap64(VAL)
#elif defined(UI_COMPILER_MSVC)
        #define UI_BYTE_SWAP_INTRINSIC_2(VAL) _byteswap_ushort(VAL)
        #define UI_BYTE_SWAP_INTRINSIC_4(VAL) _byteswap_ulong(VAL)
        #define UI_BYTE_SWAP_INTRINSIC_8(VAL) _byteswap_uint64(VAL)
#endif

namespace ui::maths {

    template <std::integral T>
    static inline constexpr auto is_power_of_2(T num) noexcept -> bool {
        if (num == 0) return true;
        return (num & (num - 1)) == 0;
    }

    static inline constexpr auto nearest_power_of_2(std::size_t num) noexcept -> std::size_t {
        if ((num & (num - 1)) == 0) return num;
        --num;
        num |= (num >> 1);
        num |= (num >> 2);
        num |= (num >> 4);
        num |= (num >> 8);
        num |= (num >> 16);
        num |= (num >> 32);
        ++num;
        return num;
    }

    static inline constexpr auto round_toward_multiple_of_2(std::size_t num) noexcept -> std::size_t {
        return num + (num & 1);
    }

    template <typename T>
        requires (std::has_unique_object_representations_v<T>)
    UI_ALWAYS_INLINE static constexpr auto byteswap(T val) noexcept {
        if constexpr (sizeof(T) == 1) return val;
        else if constexpr (sizeof(T) == 2) {
            #ifdef UI_BYTE_SWAP_INTRINSIC_2
            return UI_BYTE_SWAP_INTRINSIC_2(val);
            #else
            auto temp = std::bit_cast<std::uint16_t>(val);
            return std::bit_cast<T>((temp >> 8) | (temp << 8));
            #endif
        }
        else if constexpr (sizeof(T) == 4) {
            #ifdef UI_BYTE_SWAP_INTRINSIC_4
            return UI_BYTE_SWAP_INTRINSIC_4(val);
            #else
            auto temp = std::bit_cast<std::uint32_t>(val);
            auto s1 = (temp >> 16) | (temp << 16);
            auto upper = byteswap(static_cast<std::uint32_t>(temp >> 16)) << 16;
            auto lower = byteswap(static_cast<std::uint32_t>(temp));
            return std::bit_cast<T>(
                upper | lower
            );
            #endif
        }
        else if constexpr (sizeof(T) == 8) {
            #ifdef UI_BYTE_SWAP_INTRINSIC_8
            return UI_BYTE_SWAP_INTRINSIC_8(val);
            #else
            auto temp = std::bit_cast<std::uint64_t>(val);
            auto s1 = (temp >> 32) | (temp << 32);
            auto upper = byteswap(static_cast<std::uint32_t>(temp >> 32)) << 32;
            auto lower = byteswap(static_cast<std::uint32_t>(temp));
            return std::bit_cast<T>(upper | lower);
            #endif
        } else {
            struct Wrapper {
                std::byte data[sizeof(T)];
            };
            auto temp = std::bit_cast<Wrapper>(val);
            std::reverse(temp.begin(), temp.end());
            return std::bit_cast<T>(temp);
        }
    }

    namespace internal {
        static constexpr auto bit_reverse_map = []{
            std::array<std::uint8_t, 256> arr{};
            for (auto i = 0u; i < 256u; ++i) {
                arr[i] = static_cast<std::uint8_t>((i & 1) * (1 << (8 - 1)) | (arr[i>>1] >> 1));
            }
            return arr;
        }();
    } // namespace internal

    template <typename T>
    constexpr auto bit_reverse(T val) noexcept -> T {
        auto temp = byteswap(val);
        static constexpr auto Bytes = sizeof(T) / sizeof(std::uint8_t);
        struct Wrapper {
            std::uint8_t data[Bytes];
        };

        auto res = std::bit_cast<Wrapper>(temp);
        for (auto i = 0u; i < Bytes; ++i) {
            res.data[i] = static_cast<std::uint8_t>(internal::bit_reverse_map[res.data[i]]);
        }
        return std::bit_cast<T>(res);
    }

} // namespace ui::maths

#undef UI_BYTE_SWAP_INTRINSIC_2
#undef UI_BYTE_SWAP_INTRINSIC_4
#undef UI_BYTE_SWAP_INTRINSIC_8

#endif // AMT_UI_MATHS_HPP

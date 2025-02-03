#ifndef AMT_UI_MATHS_HPP
#define AMT_UI_MATHS_HPP

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
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

    template <typename T>
        requires std::is_arithmetic_v<T>
    static inline constexpr auto is_power_of_2(T num) noexcept -> bool {
        if (num == 0) return true;
        return (num & (num - 1)) == 0;
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

    template <typename T>
    struct FloatingPointRep;

    template <>
    struct FloatingPointRep<float> {
        bool sign;
        std::uint32_t exponent;
        std::uint32_t mantissa;
    };

    template <>
    struct FloatingPointRep<double> {
        bool sign;
        std::uint32_t exponent;
        std::uint64_t mantissa;
    };

    UI_ALWAYS_INLINE static constexpr auto decompose_fp(float n) noexcept -> FloatingPointRep<float> {
        auto bits = std::bit_cast<std::uint32_t>(n);
        if constexpr (std::endian::native == std::endian::big) {
            bits = byteswap(bits);
        }

        return {
            .sign = static_cast<bool>(bits >> 31),
            .exponent = ((bits >> 23) & 0xFF) - 127,
            .mantissa = bits & 0x7FFFFF,
        };
    }

    UI_ALWAYS_INLINE static constexpr auto decompose_fp(double n) noexcept -> FloatingPointRep<double> {
        auto bits = std::bit_cast<std::uint64_t>(n);
        if constexpr (std::endian::native == std::endian::big) {
            bits = byteswap(bits);
        }

        return {
            .sign = static_cast<bool>(bits >> 63),
            .exponent = static_cast<std::uint32_t>((bits >> 52) & 0x7FF) - 1023,
            .mantissa = bits & 0xFFFFFFFFFFFFF,
        };
    }
    
    UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<float> fp) noexcept -> float {
        auto bits = (static_cast<std::uint32_t>(fp.sign) << 31) | (((fp.exponent & 0xFF) + 127) << 23) | (fp.mantissa);
        
        if constexpr (std::endian::native == std::endian::big) {
            bits = byteswap(bits);
        }
        return std::bit_cast<float>(bits);
    }

    UI_ALWAYS_INLINE static constexpr auto compose_fp(FloatingPointRep<double> fp) noexcept -> double {
        auto bits = (static_cast<std::uint64_t>(fp.sign) << 63) | (((fp.exponent & 0x7FF) + 1023ull) << 52) | (fp.mantissa);
        
        if constexpr (std::endian::native == std::endian::big) {
            bits = byteswap(bits);
        }
        return std::bit_cast<double>(bits);
    }
} // namespace ui::maths

#undef UI_BYTE_SWAP_INTRINSIC_2
#undef UI_BYTE_SWAP_INTRINSIC_4
#undef UI_BYTE_SWAP_INTRINSIC_8

#endif // AMT_UI_MATHS_HPP

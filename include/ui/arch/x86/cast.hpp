#ifndef AMT_UI_ARC_X86_HPP
#define AMT_UI_ARC_X86_HPP

#include "../../base_vec.hpp"
#include "../../base.hpp"
#include "../basic.hpp"
#include "../../vec_headers.hpp"
#include "../../float.hpp"
#include "../../matrix.hpp"
#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>

//            #define UI_CPU_SSE_LEVEL_SSE1     10
//            #define UI_CPU_SSE_LEVEL_SSE2     20
//            #define UI_CPU_SSE_LEVEL_SSE3     30
//            #define UI_CPU_SSE_LEVEL_SSSE3    31
//            #define UI_CPU_SSE_LEVEL_SSE41    41
//            #define UI_CPU_SSE_LEVEL_SSE42    42
//            #define UI_CPU_SSE_LEVEL_AVX      51
//            #define UI_CPU_SSE_LEVEL_AVX2     52
//            #define UI_CPU_SSE_LEVEL_SKX      60
namespace ui::x86 {

	template <std::size_t N, typename T>
	UI_ALWAYS_INLINE constexpr auto to_vec(Vec<N, T> const& v) noexcept {
		if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return std::bit_cast<__m128>(v);
                } else if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256>(v);
                    #endif
                } else if constexpr (N == 16) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512>(v);
                    #endif
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                    return std::bit_cast<__m128d>(v);
                    #endif
                } else if constexpr (N == 4) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256d>(v);
                    #endif
                } else if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512d>(v);
                    #endif
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                    return std::bit_cast<__m128i>(v);
                    #endif
                } else if constexpr (N == 16) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256i>(v);
                    #endif
                } else if constexpr (N == 32) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512i>(v);
                    #endif
                }
            } else if constexpr (std::same_as<T, bfloat16>) {
                if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                    return std::bit_cast<__m128i>(v);
                    #endif
                } else if constexpr (N == 16) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256i>(v);
                    #endif
                } else if constexpr (N == 32) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512i>(v);
                    #endif
                }
            } else {
                static_assert(sizeof(T) == sizeof(float) || sizeof(T) == sizeof(double) || sizeof(T) == sizeof(float16) || sizeof(T) == sizeof(bfloat16), "Unknow floating-point type, expecting 'float', 'ui::float16', 'ui::bfloat16' or 'double'");
            }
        } else {
            static constexpr auto bits = N * sizeof(T) * 8;
            if constexpr (bits == 128) {
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                return std::bit_cast<__m128i>(v);
                #endif
            } else if constexpr (bits == 256) {
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                return std::bit_cast<__m256i>(v);
                #endif
            } else if constexpr (bits == 512) {
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                return std::bit_cast<__m512i>(v);
                #endif
            }
        } 
	}

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
    template <std::size_t N, typename T>
        requires (N * sizeof(T) == sizeof(__m128i))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m128i v) noexcept -> Vec<N, T> {
        static_assert(std::integral<T> || std::same_as<T, float16> || std::same_as<T, bfloat16>, "cannot convvert to unreleated types");
        return std::bit_cast<Vec<N, T>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
    template <std::size_t N, typename T>
        requires (N * sizeof(T) == sizeof(__m256i))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m256i v) noexcept -> Vec<N, T> {
        static_assert(std::integral<T> || std::same_as<T, float16> || std::same_as<T, bfloat16>, "cannot convvert to unreleated types");
        return std::bit_cast<Vec<N, T>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
    template <std::size_t N, typename T>
        requires (N * sizeof(T) == sizeof(__m512i))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m512i v) noexcept -> Vec<N, T> {
        static_assert(std::integral<T> || std::same_as<T, float16> || std::same_as<T, bfloat16>, "cannot convvert to unreleated types");
        return std::bit_cast<Vec<N, T>>(v); 
    }
    #endif

    template <std::size_t N, std::same_as<float> T>
        requires (N * sizeof(T) == sizeof(__m128))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m128 v) noexcept -> Vec<N, T> {
        return std::bit_cast<Vec<N, T>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m128 v) noexcept -> Vec<4, float> {
        return std::bit_cast<Vec<4, float>>(v); 
    }

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
    template <std::size_t N, std::same_as<float> T>
        requires (N * sizeof(T) == sizeof(__m256))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m256 v) noexcept -> Vec<N, float> {
        return std::bit_cast<Vec<N, float>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m256 v) noexcept -> Vec<8, float> {
        return std::bit_cast<Vec<8, float>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
    template <std::size_t N, std::same_as<float> T>
        requires (N * sizeof(T) == sizeof(__m512))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m512 v) noexcept -> Vec<N, float> {
        return std::bit_cast<Vec<N, float>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m512 v) noexcept -> Vec<16, float> {
        return std::bit_cast<Vec<16, float>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
    template <std::size_t N, std::same_as<double> T>
        requires (N * sizeof(T) == sizeof(__m128d))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m128d v) noexcept -> Vec<N, T> {
        return std::bit_cast<Vec<N, T>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m128d v) noexcept -> Vec<2, double> {
        return std::bit_cast<Vec<2, double>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
    template <std::size_t N, std::same_as<double> T>
        requires (N * sizeof(T) == sizeof(__m256d))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m256d v) noexcept -> Vec<N, double> {
        return std::bit_cast<Vec<N, double>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m256d v) noexcept -> Vec<4, double> {
        return std::bit_cast<Vec<4, double>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
    template <std::size_t N, std::same_as<double> T>
        requires (N * sizeof(T) == sizeof(__m512d))
    UI_ALWAYS_INLINE constexpr auto from_vec(__m512d v) noexcept -> Vec<N, double> {
        return std::bit_cast<Vec<N, double>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m512d v) noexcept -> Vec<8, double> {
        return std::bit_cast<Vec<8, double>>(v); 
    }
    #endif
} // namespace ui:x86

#endif // AMT_UI_ARC_X86_HPP

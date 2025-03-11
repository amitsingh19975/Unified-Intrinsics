#ifndef AMT_UI_ARCH_X86_ROUNDING_HPP
#define AMT_UI_ARCH_X86_ROUNDING_HPP

#include "cast.hpp"
#include "../emul/rounding.hpp"
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::x86 {

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <std::float_round_style mode = std::float_round_style::round_to_nearest, bool Merge = true, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto round(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::round<mode>(v);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            _mm_round_ps(to_vec(v), _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            _mm_round_ps(to_vec(v), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            _mm_round_ps(to_vec(v), _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            _mm_round_ps(to_vec(v), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)
                        );
                    } else {
                        return from_vec<T>(
                            _mm_round_ps(to_vec(v), _MM_FROUND_CUR_DIRECTION |_MM_FROUND_NO_EXC)
                        );
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            _mm_round_pd(to_vec(v), _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            _mm_round_pd(to_vec(v), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            _mm_round_pd(to_vec(v), _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            _mm_round_pd(to_vec(v), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)
                        );
                    } else {
                        return from_vec<T>(
                            _mm_round_pd(to_vec(v), _MM_FROUND_CUR_DIRECTION |_MM_FROUND_NO_EXC)
                        );
                    }
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(round<mode>(cast<float>(v)));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                return round<mode>(from_vec<T>(fit_to_vec(v))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (bits == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            _mm256_round_ps(to_vec(v), _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            _mm256_round_ps(to_vec(v), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            _mm256_round_ps(to_vec(v), _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            _mm256_round_ps(to_vec(v), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)
                        );
                    } else {
                        return from_vec<T>(
                            _mm256_round_ps(to_vec(v), _MM_FROUND_CUR_DIRECTION |_MM_FROUND_NO_EXC)
                        );
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            _mm256_round_pd(to_vec(v), _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            _mm256_round_pd(to_vec(v), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            _mm256_round_pd(to_vec(v), _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            _mm256_round_pd(to_vec(v), _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)
                        );
                    } else {
                        return from_vec<T>(
                            _mm256_round_pd(to_vec(v), _MM_FROUND_CUR_DIRECTION |_MM_FROUND_NO_EXC)
                        );
                    }
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(round<mode>(cast<float>(v)));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            _mm512_roundscale_ps(to_vec(v), _MM_FROUND_TO_ZERO)
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            _mm512_roundscale_ps(to_vec(v), _MM_FROUND_TO_NEAREST_INT)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            _mm512_roundscale_ps(to_vec(v), _MM_FROUND_TO_POS_INF)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            _mm512_roundscale_ps(to_vec(v), _MM_FROUND_TO_NEG_INF)
                        );
                    } else {
                        return from_vec<T>(
                            _mm512_roundscale_ps(to_vec(v), _MM_FROUND_CUR_DIRECTION)
                        );
                    }
                } else if constexpr (std::same_as<T, double>) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            _mm512_roundscale_pd(to_vec(v), _MM_FROUND_TO_ZERO)
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            _mm512_roundscale_pd(to_vec(v), _MM_FROUND_TO_NEAREST_INT)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            _mm512_roundscale_pd(to_vec(v), _MM_FROUND_TO_POS_INF)
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            _mm512_roundscale_pd(to_vec(v), _MM_FROUND_TO_NEG_INF)
                        );
                    } else {
                        return from_vec<T>(
                            _mm512_roundscale_pd(to_vec(v), _MM_FROUND_CUR_DIRECTION)
                        );
                    }
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(round<mode>(cast<float>(v)));
                }
            }
            #endif

            return join(
                round<mode, false>(v.lo),
                round<mode, false>(v.hi)
            );
        }
    }
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_ROUNDING_HPP

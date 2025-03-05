#ifndef AMT_UI_ARCH_X86_MINMAX_HPP
#define AMT_UI_ARCH_X86_MINMAX_HPP

#include "cast.hpp"
#include "../emul/minmax.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::x86 {

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto max(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N <= 2) {
            return emul::max(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_max_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_max_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(max(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_max_epi8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_max_epi16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_max_epi32(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_max_epu8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_max_epu16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_max_epu32(to_vec(lhs), to_vec(rhs)));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return max(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_max_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_max_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(max(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_max_epi8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_max_epi16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_max_epi32(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_max_epu8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_max_epu16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_max_epu32(to_vec(lhs), to_vec(rhs)));
                    }
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_max_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_max_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(max(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_max_epi8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_max_epi16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_max_epi32(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_max_epu8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_max_epu16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_max_epu32(to_vec(lhs), to_vec(rhs)));
                    }
                }
            }
            #endif
            return join(
                max<false>(lhs.lo, rhs.lo),
                max<false>(lhs.hi, rhs.hi)
            );
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto min(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N <= 2) {
            return emul::min(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_min_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_min_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(min(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_min_epi8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_min_epi16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_min_epi32(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_min_epu8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_min_epu16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_min_epu32(to_vec(lhs), to_vec(rhs)));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return min(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_min_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_min_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(min(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_min_epi8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_min_epi16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_min_epi32(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_min_epu8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_min_epu16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_min_epu32(to_vec(lhs), to_vec(rhs)));
                    }
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_min_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_min_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(min(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_min_epi8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_min_epi16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_min_epi32(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_min_epu8(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_min_epu16(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_min_epu32(to_vec(lhs), to_vec(rhs)));
                    }
                }
            }
            #endif
            return join(
                min<false>(lhs.lo, rhs.lo),
                min<false>(lhs.hi, rhs.hi)
            );
        }
    }

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    /**
     * @return number-maximum avoiding "NaN"
    */
    template <bool Merge = true, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto maxnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::maxnm(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = _mm_max_ps(l, r);
                    auto mask_l = _mm_cmpunord_ps(l, l); 
                    auto mask_r = _mm_cmpunord_ps(r, r); 
                    auto mask = _mm_or_ps(mask_l, mask_r);
                    auto t0 = _mm_blendv_ps(r, l, mask_r); 
                    return from_vec<T>(
                        _mm_blendv_ps(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = _mm_max_pd(l, r);
                    auto mask_l = _mm_cmpunord_pd(l, l); 
                    auto mask_r = _mm_cmpunord_pd(r, r); 
                    auto mask = _mm_or_pd(mask_l, mask_r);
                    auto t0 = _mm_blendv_pd(r, l, mask_r); 
                    return from_vec<T>(
                        _mm_blendv_pd(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(maxnm(cast<float>(lhs), cast<float>(rhs)));
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return maxnm(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = _mm256_max_ps(l, r);
                    auto mask_l = _mm256_cmp_ps(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm256_cmp_ps(r, r, _CMP_UNORD_Q);
                    auto mask = _mm256_or_ps(mask_l, mask_r);
                    auto t0 = _mm256_blendv_ps(r, l, mask_r); 
                    return from_vec<T>(
                        _mm256_blendv_ps(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = _mm256_max_pd(l, r);
                    auto mask_l = _mm256_cmp_pd(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm256_cmp_pd(r, r, _CMP_UNORD_Q);
                    auto mask = _mm256_or_pd(mask_l, mask_r);
                    auto t0 = _mm256_blendv_pd(r, l, mask_r); 
                    return from_vec<T>(
                        _mm256_blendv_pd(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(maxnm(cast<float>(lhs), cast<float>(rhs)));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = _mm512_max_ps(l, r);
                    auto mask_l = _mm512_cmp_ps_mask(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm512_cmp_ps_mask(r, r, _CMP_UNORD_Q);
                    auto mask = mask_l | mask_r;
                    auto t0 = _mm512_mask_mov_ps(r, mask_r, l);
                    return from_vec<T>(
                        _mm512_mask_mov_ps(
                            mx,
                            mask,
                            t0
                        )
                    );
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = _mm512_max_pd(l, r);
                    auto mask_l = _mm512_cmp_pd_mask(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm512_cmp_pd_mask(r, r, _CMP_UNORD_Q);
                    auto mask = mask_l | mask_r;
                    auto t0 = _mm512_mask_mov_pd(r, mask_r, l);
                    return from_vec<T>(
                        _mm512_mask_mov_pd(
                            mx,
                            mask,
                            t0
                        )
                    );
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(maxnm(cast<float>(lhs), cast<float>(rhs)));
                }
            }
            #endif
            return join(
                maxnm<false>(lhs.lo, rhs.lo),
                maxnm<false>(lhs.hi, rhs.hi)
            );
        }
    }

    /**
     * @return number-minimum avoiding "NaN"
    */
    template <bool Merge = true, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto minnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::minnm(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = _mm_min_ps(l, r);
                    auto mask_l = _mm_cmpunord_ps(l, l); 
                    auto mask_r = _mm_cmpunord_ps(r, r); 
                    auto mask = _mm_or_ps(mask_l, mask_r);
                    auto t0 = _mm_blendv_ps(r, l, mask_r); 
                    return from_vec<T>(
                        _mm_blendv_ps(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = _mm_min_pd(l, r);
                    auto mask_l = _mm_cmpunord_pd(l, l); 
                    auto mask_r = _mm_cmpunord_pd(r, r); 
                    auto mask = _mm_or_pd(mask_l, mask_r);
                    auto t0 = _mm_blendv_pd(r, l, mask_r); 
                    return from_vec<T>(
                        _mm_blendv_pd(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(minnm(cast<float>(lhs), cast<float>(rhs)));
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return minnm(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = _mm256_min_ps(l, r);
                    auto mask_l = _mm256_cmp_ps(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm256_cmp_ps(r, r, _CMP_UNORD_Q);
                    auto mask = _mm256_or_ps(mask_l, mask_r);
                    auto t0 = _mm256_blendv_ps(r, l, mask_r); 
                    return from_vec<T>(
                        _mm256_blendv_ps(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = _mm256_min_pd(l, r);
                    auto mask_l = _mm256_cmp_pd(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm256_cmp_pd(r, r, _CMP_UNORD_Q);
                    auto mask = _mm256_or_pd(mask_l, mask_r);
                    auto t0 = _mm256_blendv_pd(r, l, mask_r); 
                    return from_vec<T>(
                        _mm256_blendv_pd(
                            mx,
                            t0,
                            mask
                        )
                    );
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(minnm(cast<float>(lhs), cast<float>(rhs)));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = _mm512_min_ps(l, r);
                    auto mask_l = _mm512_cmp_ps_mask(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm512_cmp_ps_mask(r, r, _CMP_UNORD_Q);
                    auto mask = mask_l | mask_r;
                    auto t0 = _mm512_mask_mov_ps(r, mask_r, l);
                    return from_vec<T>(
                        _mm512_mask_mov_ps(
                            mx,
                            mask,
                            t0
                        )
                    );
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = _mm512_min_pd(l, r);
                    auto mask_l = _mm512_cmp_pd_mask(l, l, _CMP_UNORD_Q);
                    auto mask_r = _mm512_cmp_pd_mask(r, r, _CMP_UNORD_Q);
                    auto mask = mask_l | mask_r;
                    auto t0 = _mm512_mask_mov_pd(r, mask_r, l);
                    return from_vec<T>(
                        _mm512_mask_mov_pd(
                            mx,
                            mask,
                            t0
                        )
                    );
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(minnm(cast<float>(lhs), cast<float>(rhs)));
                }
            }
            #endif
            return join(
                minnm<false>(lhs.lo, rhs.lo),
                minnm<false>(lhs.hi, rhs.hi)
            );
        }
    }

// MARK: Pairwise Maximum
    namespace internal {
        template <bool Merge = true, std::size_t N, typename T>
            requires (N > 1)
        UI_ALWAYS_INLINE auto pminmax_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            auto&& fn
        ) noexcept -> Vec<N, T> {
            static constexpr auto size = sizeof(lhs);
            if constexpr (N <= 2) {
                return emul::pmax(lhs, rhs);
            } else {
                if constexpr (size == sizeof(__m128)) {
                    auto l = to_vec(lhs);
                    auto r = to_vec(rhs);
                    if constexpr (std::same_as<T, float>) {
                        // pminmax_helper([a0, a1, a2, a3], [b0, b1, b2, b3], max)
                        // [max(a0, a1), max(a2, a3), max(b0, b1), max(b2, b3)]
                        auto t0 = _mm_shuffle_ps(l, r, _MM_SHUFFLE(2, 0, 2, 0));
                        auto t1 = _mm_shuffle_ps(l, r, _MM_SHUFFLE(3, 1, 3, 1));

                        auto mx = fn(t0, t1);
                        return from_vec<T>(
                            mx
                        );
                    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                        return cast<T>(pminmax_helper(cast<float>(lhs), cast<float>(rhs), fn));
                    } else if constexpr (sizeof(T) == 1) {
                        static constexpr std::uint8_t odd_mask[16] = {
                            1, 1, 3, 3, 5, 5, 7, 7,
                            9, 9, 11, 11, 13, 13, 15, 15 
                        };
                        static constexpr std::uint8_t even_pos_mask[16] = {
                            0, 2, 4, 6, 8, 10, 12, 14,
                            16, 18, 20, 22, 24, 26, 28, 30
                        };

                        auto t0 = _mm_shuffle_epi8(l, *reinterpret_cast<__m128i const*>(odd_mask)); 
                        auto lmx = fn(t0, l);
                        auto lo = _mm_shuffle_epi8(lmx, *reinterpret_cast<__m128i const*>(even_pos_mask)); 

                        auto t1 = _mm_shuffle_epi8(r, *reinterpret_cast<__m128i const*>(odd_mask)); 
                        auto rmx = fn(t1, r);
                        auto hi = _mm_shuffle_epi8(rmx, *reinterpret_cast<__m128i const*>(even_pos_mask)); 

                        return from_vec<T>(_mm_unpacklo_epi64(lo, hi));
                    } else if constexpr (sizeof(T) == 2) {
                        // [a00, a01, a10, a11, a20, a21, a30, a31, ...]
                        static constexpr std::uint8_t odd_mask[16] = {
                             2,  3,  2,  3,  6,  7,  6,  7,
                            10, 11, 10, 11, 14, 15, 14, 15
                        };
                        static constexpr std::uint8_t even_pos_mask[16] = {
                             0,  1,  4,  5,  8,  9, 12, 13,
                            16, 17, 20, 21, 24, 26, 30, 31
                        };

                        auto t0 = _mm_shuffle_epi8(l, *reinterpret_cast<__m128i const*>(odd_mask)); 
                        auto lmx = fn(t0, l);
                        auto lo = _mm_shuffle_epi8(lmx, *reinterpret_cast<__m128i const*>(even_pos_mask)); 

                        auto t1 = _mm_shuffle_epi8(r, *reinterpret_cast<__m128i const*>(odd_mask)); 
                        auto rmx = fn(t1, r);
                        auto hi = _mm_shuffle_epi8(rmx, *reinterpret_cast<__m128i const*>(even_pos_mask)); 

                        return from_vec<T>(_mm_unpacklo_epi64(lo, hi));
                    } else if constexpr (sizeof(T) == 4) {
                        auto t0 = _mm_shuffle_epi32(l, _MM_SHUFFLE(3, 1, 3, 1));
                        auto t1 = _mm_shuffle_epi32(r, _MM_SHUFFLE(3, 1, 3, 1));

                        auto lo = fn(t0, l);
                        auto hi = fn(t1, r);
                        return from_vec<T>(_mm_unpacklo_epi64(lo, hi));
                    }
                } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                    return pminmax_helper(
                        from_vec<T>(fit_to_vec(lhs)),
                        from_vec<T>(fit_to_vec(rhs)),
                        fn
                    ).lo;
                }

                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (size == sizeof(__m256)) {
                    auto l = to_vec(lhs);
                    auto r = to_vec(rhs);
                    if constexpr (std::same_as<T, float>) {
                        auto t0 = _mm256_shuffle_ps(l, r, _MM_SHUFFLE(2,0,2,0));
                        auto t1 = _mm256_shuffle_ps(l, r, _MM_SHUFFLE(3,1,3,1));

                        auto t0d = _mm256_castps_pd(t0);
                        auto t1d = _mm256_castps_pd(t1);

                        auto t0d_perm = _mm256_permute4x64_pd(t0d, 0b11'01'10'00); // [3, 1, 2, 0]
                        auto t1d_perm = _mm256_permute4x64_pd(t1d, 0b11'01'10'00);

                        auto t0_perm = _mm256_castpd_ps(t0d_perm);
                        auto t1_perm = _mm256_castpd_ps(t1d_perm);

                        auto mx = fn(t0_perm, t1_perm);
                        return from_vec<T>(mx);
                    } else if constexpr (std::same_as<T, double>) {
                        // INFO: This is a direct translation of clang output
                        auto t0 = _mm256_permute2f128_pd(l, r, 0x31);
                        auto t1 = _mm256_permute2f128_pd(l, r, 0x20);

                        auto t_low = _mm256_unpacklo_pd(t1, t0);
                        auto t_high = _mm256_unpackhi_pd(t1, t0);

                        return from_vec<T>(fn(t_low, t_high));
                    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                        return cast<T>(pminmax_helper(cast<float>(lhs), cast<float>(rhs), fn));
                    } else if constexpr (sizeof(T) == 1) {
                        static constexpr std::uint8_t even_mask[32] = {
                            0, 2, 4, 6, 8, 10, 12, 14,
                            0, 2, 4, 6, 8, 10, 12, 14,
                            0, 2, 4, 6, 8, 10, 12, 14,
                            0, 2, 4, 6, 8, 10, 12, 14
                        };
                        static constexpr std::uint8_t odd_mask[32] = {
                            1, 3, 5, 7, 9, 11, 13, 15,
                            1, 3, 5, 7, 9, 11, 13, 15,
                            1, 3, 5, 7, 9, 11, 13, 15,
                            1, 3, 5, 7, 9, 11, 13, 15
                        };

                        auto le = _mm256_shuffle_epi8(l, *reinterpret_cast<__m256i const*>(even_mask)); 
                        auto re = _mm256_shuffle_epi8(r, *reinterpret_cast<__m256i const*>(even_mask)); 

                        auto b_even = _mm256_blend_epi32(le, re, 0b1100'1100);
                        auto even = _mm256_permute4x64_epi64(b_even, 0b11'01'10'00);

                        auto lo = _mm256_shuffle_epi8(l, *reinterpret_cast<__m256i const*>(odd_mask)); 
                        auto ro = _mm256_shuffle_epi8(r, *reinterpret_cast<__m256i const*>(odd_mask)); 

                        auto b_odd = _mm256_blend_epi32(lo, ro, 0b1100'1100);
                        auto odd = _mm256_permute4x64_epi64(b_odd, 0b11'01'10'00);

                        return from_vec<T>(fn(even, odd));
                    } else if constexpr (sizeof(T) == 2) {
                            // INFO: This is a direct translation of clang output
                            static const int32_t even_mask[4] = {
                                84148480, 218892552, 353636624, 488380696
                            };
                            auto temp = *reinterpret_cast<__m128i const*>(even_mask);
                            auto mask2 = _mm256_cvtepi32_epi64(temp);

                            auto shuf_l = _mm256_shuffle_epi8(l, mask2);
                            auto shuf_r = _mm256_shuffle_epi8(r, mask2);
                            auto shuf_l_ps = _mm256_castsi256_ps(shuf_l);
                            auto shuf_r_ps = _mm256_castsi256_ps(shuf_r);
                            auto t0 = _mm256_shuffle_ps(shuf_l_ps, shuf_r_ps, _MM_SHUFFLE(2,0,2,0));
                            auto t0_perm = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(t0), 216));

                            static const uint8_t odd_mask[32] = {
                                 2,  3,  6,  7, 10, 11, 14, 15,
                                 2,  3,  6,  7, 10, 11, 14, 15,
                                18, 19, 22, 23, 26, 27, 30, 31,
                                18, 19, 22, 23, 26, 27, 30, 31
                            };
                            auto mask1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(odd_mask));

                            auto shuf_l2 = _mm256_shuffle_epi8(l, mask1);
                            auto shuf_r2 = _mm256_shuffle_epi8(r, mask1);
                            auto blended = _mm256_blend_epi32(shuf_l2, shuf_r2, 0xCC);
                            auto blended_perm = _mm256_permute4x64_epi64(blended, 216);

                            auto part1 = _mm256_castps_si256(t0_perm);
                            return from_vec<T>(fn(part1, blended_perm));
                    } else if constexpr (sizeof(T) == 4) {
                        // INFO: This is a direct translation of clang output
                        auto l_ps = _mm256_castsi256_ps(l);
                        auto r_ps = _mm256_castsi256_ps(r);

                        auto t0 = _mm256_shuffle_ps(l_ps, r_ps, _MM_SHUFFLE(2,0,2,0));
                        auto t1 = _mm256_shuffle_ps(l_ps, r_ps, _MM_SHUFFLE(3,1,3,1));

                        t0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(t0), 216));
                        t1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(t1), 216));

                        auto t0i = _mm256_castps_si256(t0);
                        auto t1i = _mm256_castps_si256(t1);

                        return from_vec<T>(fn(t0i, t1i));
                    } else if constexpr (sizeof(T) == 8) {
                        // INFO: This is a direct translation of clang output
                        auto t_low = _mm256_unpacklo_epi64(l, r);
                        auto t_low_perm = _mm256_permute4x64_epi64(t_low, 216);

                        auto t_high = _mm256_unpackhi_epi64(l, r);
                        auto t_high_perm = _mm256_permute4x64_epi64(t_high, 216);

                        __m256i cmp_mask;

                        if constexpr (std::is_signed_v<T>) {
                            cmp_mask = fn(t_low_perm, t_high_perm);
                        } else {
                            auto bias = _mm256_set1_epi64x(static_cast<std::int64_t>(0x8000000000000000ULL));
                            auto l_biased = _mm256_xor_si256(t_low_perm, bias);
                            auto r_biased = _mm256_xor_si256(t_high_perm, bias);
                            cmp_mask = fn(l_biased, r_biased);
                        }

                        auto t_low_pd  = _mm256_castsi256_pd(t_low_perm);
                        auto t_high_pd = _mm256_castsi256_pd(t_high_perm);
                        auto mask_pd   = _mm256_castsi256_pd(cmp_mask);

                        auto blended_pd = _mm256_blendv_pd(t_high_pd, t_low_pd, mask_pd);

                        return from_vec<T>(_mm256_castpd_si256(blended_pd));
                    }
                }
                #endif

                // TODO: Add avx 512 implementation
                return join(
                    pminmax_helper<false>(lhs.lo, lhs.hi, fn),
                    pminmax_helper<false>(rhs.lo, rhs.hi, fn)
                );
            }
        }

        template <bool Merge = true, typename O, std::size_t N, typename T>
            requires (N > 1)
        UI_ALWAYS_INLINE auto fold_helper(
            Vec<N, T> const& v,
            O op
        ) noexcept -> T {
            static constexpr auto size = sizeof(v);
            static constexpr auto is_max = std::is_same_v<O, op::pmax_t> || std::is_same_v<O, op::pmaxnm_t>;
            constexpr auto fn = [](auto x, auto y) {
                if constexpr (std::same_as<O, op::pmax_t>) {
                    return to_vec(max(from_vec<T>(x), from_vec<T>(y)));
                } else if constexpr (std::same_as<O, op::pmaxnm_t>) {
                    return to_vec(maxnm(from_vec<T>(x), from_vec<T>(y)));
                } else if constexpr (std::same_as<O, op::pmin_t>) {
                    return to_vec(min(from_vec<T>(x), from_vec<T>(y)));
                } else if constexpr (std::same_as<O, op::pminnm_t>) {
                    return to_vec(minnm(from_vec<T>(x), from_vec<T>(y)));
                }
            };
            if constexpr (N == 2) {
                return emul::fold(v, op);
            } else {
                if constexpr (size == sizeof(__m128)) {
                    auto a = to_vec(v);
                    if constexpr (std::same_as<T, float>) {
                        auto s0 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 2, 0));
                        auto s1 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 3, 1));

                        auto t0 = fn(s0, s1);
                        auto t1 = _mm_movehdup_ps(t0);
                        auto mx = fn(t0, t1);
                        return from_vec<T>(mx)[0];
                    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                        return fold_helper(cast<float>(v));
                    } else if constexpr (sizeof(T) == 1) {
                        if constexpr (is_max) {
                            if constexpr (std::is_signed_v<T>) {
                                auto mask = _mm_set1_epi8(static_cast<std::int8_t>(0x7f));
                                auto t0 = _mm_xor_si128(a, mask); // get rid of sign
                                auto t1 = _mm_srli_epi16(t0, 8);
                                auto t2 = _mm_min_epu8(t0, t1);
                                auto t3 = _mm_minpos_epu16(t2);
                                return static_cast<T>(_mm_cvtsi128_si32(t3) ^ 0x7f);
                            } else {
                                auto t0 = _mm_set1_epi32(~std::int32_t{});
                                auto t1 = _mm_xor_si128(t0, a);
                                auto t2 = _mm_srli_epi16(t1, 8);
                                auto t3 = _mm_min_epu8(t1, t2);
                                auto t4 = _mm_minpos_epu16(t3);
                                return static_cast<T>(~_mm_cvtsi128_si32(t4));
                            }
                        } else {
                            if constexpr (std::is_signed_v<T>) {
                                auto mask = _mm_set1_epi8(static_cast<std::int8_t>(0x80));
                                auto t0 = _mm_xor_si128(a, mask); // get rid of sign
                                auto t1 = _mm_srli_epi16(t0, 8); // logical shift right by 8
                                auto t2 = _mm_min_epu8(t0, t1);
                                auto t3 = _mm_minpos_epu16(t2);
                                return static_cast<T>(_mm_cvtsi128_si32(t3) - 128);
                            } else {
                                auto t1 = _mm_srli_epi16(a, 8); // logical shift right by 8
                                auto t2 = _mm_min_epu8(a, t1);
                                auto t3 = _mm_minpos_epu16(t2);
                                return static_cast<T>(_mm_cvtsi128_si32(t2));
                            }
                        }
                    } else if constexpr (sizeof(T) == 2) {
                        if constexpr (is_max) {
                            if constexpr (std::is_signed_v<T>) {
                                auto mask = _mm_set1_epi16(static_cast<std::int16_t>(0x7fff));
                                auto t0 = _mm_xor_si128(mask, a);
                                auto t1 = _mm_minpos_epu16(t0);
                                return static_cast<T>(_mm_cvtsi128_si32(t1) ^ 0x7f'ff);
                            } else {
                                auto mask = _mm_set1_epi32(~std::int32_t{});
                                auto t0 = _mm_xor_si128(mask, a);
                                auto t1 = _mm_minpos_epu16(t0);
                                return static_cast<T>(~_mm_cvtsi128_si32(t1));
                            }
                        } else {
                            if constexpr (std::is_signed_v<T>) {
                                auto mask = _mm_set1_epi16(0x80'00);
                                auto t0 = _mm_xor_si128(mask, a);
                                auto t1 = _mm_minpos_epu16(t0);
                                return static_cast<T>(_mm_cvtsi128_si32(t1) ^ 0x80'00);
                            } else {
                                auto t0 = _mm_minpos_epu16(a);
                                return static_cast<T>(_mm_cvtsi128_si32(t0));
                            }
                        }
                    } else if constexpr (sizeof(T) == 4) {
                        auto b = to_vec(join(v.hi, v.hi));
                        auto mx = fn(a, b);
                        auto s = _mm_shuffle_epi32(mx, _MM_SHUFFLE(1, 1, 1, 1));
                        mx = fn(mx, s);
                        return static_cast<T>(_mm_cvtsi128_si32(mx));
                    }
                } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                    return fold_helper(
                        join(v, v), op
                    );
                }

                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (size == sizeof(__m256)) {
                    auto a = to_vec(v);
                    if constexpr (std::same_as<T, float>) {
                        auto t0 = _mm256_permute_ps(a, _MM_SHUFFLE(2, 3, 0, 1)); 
                        auto t1 = fn(t0, a);
                        auto t2 = _mm256_permute_ps(t1, _MM_SHUFFLE(0, 0, 2, 2));
                        auto t3 = fn(t1, t2);

                        auto t4 = _mm256_extractf128_ps(t3, 1);
                        auto t5 = fn(_mm256_castps256_ps128(t3), t4);
                        return from_vec<T>(t5)[0];
                    } else if constexpr (std::same_as<T, double>) {
                        auto t0 = _mm256_permute2f128_pd(a, a, 1);
                        auto t1 = fn(t0, a);
                        auto t2 = _mm256_permute_pd(t1, 5);
                        return from_vec<T>(fn(t1, t2))[0];
                    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                        return fold_helper(cast<float>(v));
                    } else if constexpr (sizeof(T) == 1) {
                        auto lo = _mm256_castsi256_si128(a);
                        auto hi = _mm256_extracti128_si256(a, 1);
                        auto t = fn(lo, hi);
                        return fold_helper<false>(from_vec<T>(t), op);
                    } else if constexpr (sizeof(T) == 2) {
                        auto lo = _mm256_castsi256_si128(a);
                        auto hi = _mm256_extracti128_si256(a, 1);
                        auto t = fn(lo, hi);
                        return fold_helper<false>(from_vec<T>(t), op);
                    } else if constexpr (sizeof(T) == 4) {
                        auto lo = _mm256_castsi256_si128(a);
                        auto hi = _mm256_extracti128_si256(a, 1);
                        auto t = fn(lo, hi);
                        return fold_helper<false>(from_vec<T>(t), op);
                    }
                }
                #endif

                // TODO: Add avx 512 implementation
                if constexpr (is_max) {
                    if constexpr (std::same_as<O, op::pmaxnm_t>) {
                        return internal::maxnm(
                            fold_helper<false>(v.lo, op),
                            fold_helper<false>(v.hi, op)
                        );
                    } else {
                        return internal::max(
                            fold_helper<false>(v.lo, op),
                            fold_helper<false>(v.hi, op)
                        );
                    }
                } else {
                    if constexpr (std::same_as<O, op::pminnm_t>) {
                        return internal::minnm(
                            fold_helper<false>(v.lo, op),
                            fold_helper<false>(v.hi, op)
                        );
                    } else {
                        return internal::min(
                            fold_helper<false>(v.lo, op),
                            fold_helper<false>(v.hi, op)
                        );
                    }
                }
            }
        }
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmax(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(lhs, rhs, [](auto l, auto r) {
            if constexpr (sizeof(l) == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) return _mm_max_ps(l, r);
                else if constexpr (std::same_as<T, double>) return _mm_max_pd(l, r);
                else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) return _mm_max_epi8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm_max_epi16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm_max_epi32(l, r);
                    else return _mm_cmpgt_epi64(l, r);
                } else {
                    if constexpr (sizeof(T) == 1) return _mm_max_epu8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm_max_epu16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm_max_epu32(l, r);
                    else return _mm_cmpgt_epi64(l, r);
                }
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (sizeof(l) == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) return _mm256_max_ps(l, r);
                else if constexpr (std::same_as<T, double>) return _mm256_max_pd(l, r);
                else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) return _mm256_max_epi8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm256_max_epi16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm256_max_epi32(l, r);
                    else return _mm256_cmpgt_epi64(l, r);
                } else {
                    if constexpr (sizeof(T) == 1) return _mm256_max_epu8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm256_max_epu16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm256_max_epu32(l, r);
                    else return _mm256_cmpgt_epi64(l, r);
                }
            }
            #endif
        });
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmax_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::max_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pmax_t{});
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmaxnm(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(x, y, [](auto l, auto r) {
            return to_vec(maxnm(from_vec<T>(l), from_vec<T>(r)));
        });
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmaxnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::maxnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pmaxnm_t{});
    }
// !MARK

// MARK: Pairwise Minimum
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmin(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(lhs, rhs, [](auto l, auto r) {
            if constexpr (sizeof(l) == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) return _mm_min_ps(l, r);
                else if constexpr (std::same_as<T, double>) return _mm_min_pd(l, r);
                else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) return _mm_min_epi8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm_min_epi16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm_min_epi32(l, r);
                    else return _mm_cmpgt_epi64(r, l);
                } else {
                    if constexpr (sizeof(T) == 1) return _mm_min_epu8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm_min_epu16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm_min_epu32(l, r);
                    else return _mm_cmpgt_epi64(r, l);
                }
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (sizeof(l) == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) return _mm256_min_ps(l, r);
                else if constexpr (std::same_as<T, double>) return _mm256_min_pd(l, r);
                else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) return _mm256_min_epi8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm256_min_epi16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm256_min_epi32(l, r);
                    else return _mm256_cmpgt_epi64(r, l);
                } else {
                    if constexpr (sizeof(T) == 1) return _mm256_min_epu8(l, r);
                    else if constexpr (sizeof(T) == 2) return _mm256_min_epu16(l, r);
                    else if constexpr (sizeof(T) == 4) return _mm256_min_epu32(l, r);
                    else return _mm256_cmpgt_epi64(r, l);
                }
            }
            #endif
        });
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmin_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::min_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pmin_t{});
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pminnm(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(x, y, [](auto l, auto r) {
            return to_vec(minnm(from_vec<T>(l), from_vec<T>(r)));
        });
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pminnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::minnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pminnm_t{});
    }
// !MARK
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_MINMAX_HPP

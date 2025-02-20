#ifndef AMT_ARCH_X86_LOAD_HPP
#define AMT_ARCH_X86_LOAD_HPP

#include "cast.hpp"
#include <algorithm>
#include <utility>

namespace ui::x86 {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(T val) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(val) };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    return from_vec<T>(_mm_set1_ps(val));
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                if constexpr (N * sizeof(T) == sizeof(__m256)) {
                    return from_vec<T>(_mm256_set1_ps(val));
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (N * sizeof(T) == sizeof(__m512)) {
                    return from_vec<T>(_mm512_set1_ps(val));
                }
                #endif
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    return from_vec<T>(_mm_set1_pd(val));
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                if constexpr (N * sizeof(T) == sizeof(__m256)) {
                    return from_vec<T>(_mm256_set1_pd(val));
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (N * sizeof(T) == sizeof(__m512)) {
                    return from_vec<T>(_mm512_set1_pd(val));
                }
                #endif
            } else {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_set1_epi8(val));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_set1_epi16(val));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_set1_epi32(val));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_set1_epi64x((val)));
                    }
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                if constexpr (N * sizeof(T) == sizeof(__m256)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_set1_epi8(val));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_set1_epi16(val));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_set1_epi32(val));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_set1_epi64x(val));
                    }
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (N * sizeof(T) == sizeof(__m512)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_set1_epi8(val));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_set1_epi16(val));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_set1_epi32(val));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_set1_epi64(val));
                    }
                }
                #endif
            }
            return join(load<N / 2>(val), load<N / 2>(val));
        }
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        return load(v[Lane]);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b
    ) noexcept {
        if constexpr (N == 1) {
            a.val = data[0];
            b.val = data[1];
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_ps(data); // [d0, d1, d2, d3]
                    auto t1 = _mm_loadu_ps(data + 4); // [d4, d5, d6, d7]
                    a = from_vec<T>(_mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0))); // [d0, d1]
                    b = from_vec<T>(_mm_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 1, 3, 1))); // [d2, d3]
                    return;
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_pd(data); // [d0, d1]
                    auto t1 = _mm_loadu_pd(data + 2); // [d2, d3]
                    a.lo = from_vec<T>(_mm_shuffle_pd(t0, t0, _MM_SHUFFLE(0, 0, 2, 0))).lo; // [d0, d1]
                    b.lo = from_vec<T>(_mm_shuffle_pd(t0, t0, _MM_SHUFFLE(0, 0, 3, 1))).lo; // [d2, d3]
                    a.hi = from_vec<T>(_mm_shuffle_pd(t1, t1, _MM_SHUFFLE(0, 0, 2, 0))).lo; // [d4, d5]
                    b.hi = from_vec<T>(_mm_shuffle_pd(t1, t1, _MM_SHUFFLE(0, 0, 3, 1))).lo; // [d6, d7]
                    return;
                }   
            } else if constexpr (sizeof(T) == 1) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    alignas(16) static constexpr std::uint8_t mask[] = {
                        0,  2,  4,  6,  8, 10, 12, 14,
                        0,  0,  0,  0,  0,  0,  0,  0,
                        1,  3,  5,  7,  9, 11, 13, 15,
                        0,  0,  0,  0,  0,  0,  0,  0
                    };
                    auto even_mask = *reinterpret_cast<__m128i const*>(mask);
                    auto odd_mask = *reinterpret_cast<__m128i const*>(mask + 16);
                    a.lo = from_vec<T>(_mm_shuffle_epi8(t0, even_mask)).lo;
                    b.lo = from_vec<T>(_mm_shuffle_epi8(t0, odd_mask)).lo;
                    a.hi = from_vec<T>(_mm_shuffle_epi8(t1, even_mask)).lo;
                    b.hi = from_vec<T>(_mm_shuffle_epi8(t1, odd_mask)).lo;
                    return;
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    alignas(16) static constexpr std::uint8_t mask[] = {
                         0,  1,   // element 0
                         4,  5,   // element 2
                         8,  9,   // element 4
                        12, 13,   // element 6
                        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
                         2,  3,   // element 1
                         6,  7,   // element 3
                        10, 11,   // element 5
                        14, 15,   // element 7
                        0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
                    };
                    auto even_mask = *reinterpret_cast<__m128i const*>(mask);
                    auto odd_mask = *reinterpret_cast<__m128i const*>(mask + 16);
                    a.lo = from_vec<T>(_mm_shuffle_epi8(t0, even_mask)).lo;
                    b.lo = from_vec<T>(_mm_shuffle_epi8(t0, odd_mask)).lo;
                    a.hi = from_vec<T>(_mm_shuffle_epi8(t1, even_mask)).lo;
                    b.hi = from_vec<T>(_mm_shuffle_epi8(t1, odd_mask)).lo;
                    return;
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    a.lo = from_vec<T>(_mm_shuffle_epi32(t0, _MM_SHUFFLE(0, 0, 2, 0))).lo;
                    b.lo = from_vec<T>(_mm_shuffle_epi32(t0, _MM_SHUFFLE(0, 0, 3, 1))).lo;
                    a.hi = from_vec<T>(_mm_shuffle_epi32(t1, _MM_SHUFFLE(0, 0, 2, 0))).lo;
                    b.hi = from_vec<T>(_mm_shuffle_epi32(t1, _MM_SHUFFLE(0, 0, 3, 1))).lo;
                    return;
                }
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    a.lo = from_vec<T>(_mm_shuffle_epi32(t0, _MM_SHUFFLE(5, 4, 1, 0))).lo;
                    b.lo = from_vec<T>(_mm_shuffle_epi32(t0, _MM_SHUFFLE(7, 6, 3, 2))).lo;
                    a.hi = from_vec<T>(_mm_shuffle_epi32(t1, _MM_SHUFFLE(5, 4, 1, 0))).lo;
                    b.hi = from_vec<T>(_mm_shuffle_epi32(t1, _MM_SHUFFLE(7, 6, 3, 2))).lo;
                    return;
                }
            }
            strided_load(data, a.lo, b.lo);
            strided_load(data + N / 2 * 2, a.hi, b.hi);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c
    ) noexcept {
        using namespace constants;
        if constexpr (N == 1) {
            a.val = data[0];
            b.val = data[1];
            c.val = data[2];
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_ps(data); // [d0, d1, d2, d3]
                    auto t1 = _mm_loadu_ps(data + 4); // [d4, d5, d6, d7]
                    auto t2 = _mm_loadu_ps(data + 8);
                    auto tmp0 = _mm_castsi128_ps(
                        _mm_shuffle_epi32(
                            _mm_castps_si128(t0),
                            0b10'01'11'00 // [2, 1, 3, 0]
                        )
                    ); //a0,a3,a1,a2
                    auto tmp1 = _mm_castsi128_ps(
                        _mm_shuffle_epi32(
                            _mm_castps_si128(t1),
                            swap_hi_low32
                        )
                    ); //b2,b3,b0,b1
                    auto tmp2 = _mm_castsi128_ps(
                        _mm_shuffle_epi32(
                            _mm_castps_si128(t2),
                            0b11'00'10'01 // [3, 0, 2, 1]
                        )
                    ); //c1,c2, c0,c3
                    auto tmp3 = _mm_unpacklo_ps(tmp1, tmp2); //b2,c1, b3,c2

                    a = from_vec<T>(_mm_movelh_ps(tmp0, tmp3)); //a0,a3,b2,c1
                    tmp0 = _mm_unpackhi_ps(tmp0, tmp1); //a1,b0, a2,b1
                    b = from_vec<T>(
                        _mm_castsi128_ps(
                            _mm_shuffle_epi32(
                                _mm_castps_si128(tmp0),
                                swap_hi_low32
                            )
                        )
                    ); //a2,b1, a1,b0,
                    b = from_vec<T>(_mm_movehl_ps(tmp3, to_vec(b))); //a1,b0, b3,c2
                    c = from_vec<T>(_mm_movehl_ps(tmp2, tmp0)); //a2,b1, c0,c3
                    return;
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_pd(data); // [d0, d1]
                    auto t1 = _mm_loadu_pd(data + 2); // [d2, d3]
                    auto t2 = _mm_loadu_pd(data + 4);

                    a = from_vec<T>(_mm_shuffle_pd(t0, t1, 2));
                    b = from_vec<T>(_mm_shuffle_pd(t0, t2, 1));
                    c = from_vec<T>(_mm_shuffle_pd(t1, t2, 2));
                    return;
                }   
            } else if constexpr (sizeof(T) == 1) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);

                    alignas(16) static constexpr std::int8_t mask8_0[16] = {0,3,6,9,12,15,1,4,7,10,13,2,5,8,11,14};
                    alignas(16) static constexpr std::int8_t mask8_1[16] = {2,5,8,11,14,0,3,6,9,12,15,1,4,7,10,13};
                    alignas(16) static constexpr std::int8_t mask8_2[16] = {1,4,7,10,13,2,5,8,11,14,0,3,6,9,12,15};

                    auto tmp0 = _mm_shuffle_epi8(t0, *reinterpret_cast<__m128i const*>(mask8_0));
                    auto tmp1 = _mm_shuffle_epi8(t1, *reinterpret_cast<__m128i const*>(mask8_1));
                    auto tmp2 = _mm_shuffle_epi8(t2, *reinterpret_cast<__m128i const*>(mask8_2));

                    auto tmp3 = _mm_slli_si128(tmp0, 10); // //0,0,0,0,0,0,0,0,0,0,a0,a3,a6,a9,a12,a15
                    tmp3 = _mm_alignr_epi8(tmp1, tmp3, 10); //a:0,3,6,9,12,15,b:2,5,8,11,14,x,x,x,x,x
                    tmp3 = _mm_slli_si128(tmp3, 5); //0,0,0,0,0,a:0,3,6,9,12,15,b:2,5,8,11,14,
                    tmp3 = _mm_srli_si128(tmp3, 5); //a:0,3,6,9,12,15,b:2,5,8,11,14,:0,0,0,0,0

                    auto tmp4 = _mm_slli_si128(tmp2, 11); //0,0,0,0,0,0,0,0,0,0,0,0, 1,4,7,10,13,
                    a = from_vec<T>(_mm_or_si128(tmp4, tmp3)); //a:0,3,6,9,12,15,b:2,5,8,11,14,c:1,4,7,10,13,
                    
                    tmp3 = _mm_slli_si128(tmp0, 5); //0,0,0,0,0,a:0,3,6,9,12,15,1,4,7,10,13,
                    tmp3 = _mm_srli_si128(tmp3, 11); //a:1,4,7,10,13, 0,0,0,0,0,0,0,0,0,0,0
                    tmp4 = _mm_srli_si128(tmp1, 5); //b:0,3,6,9,12,15,C:1,4,7,10,13, 0,0,0,0,0
                    tmp4 = _mm_slli_si128(tmp4, 5); //0,0,0,0,0,b:0,3,6,9,12,15,C:1,4,7,10,13,
                    tmp4 = _mm_or_si128(tmp4, tmp3); //a:1,4,7,10,13,b:0,3,6,9,12,15,C:1,4,7,10,13,
                    tmp4 = _mm_slli_si128(tmp4, 5); //0,0,0,0,0,a:1,4,7,10,13,b:0,3,6,9,12,15,
                    tmp4 = _mm_srli_si128(tmp4, 5); //a:1,4,7,10,13,b:0,3,6,9,12,15,0,0,0,0,0
                
                    tmp3 = _mm_srli_si128(tmp2, 5); //c:2,5,8,11,14,0,3,6,9,12,15,0,0,0,0,0
                    tmp3 = _mm_slli_si128(tmp3, 11); //0,0,0,0,0,0,0,0,0,0,0,c:2,5,8,11,14,
                    b = from_vec<T>(_mm_or_si128(tmp4, tmp3)); //a:1,4,7,10,13,b:0,3,6,9,12,15,c:2,5,8,11,14,

                    tmp3 = _mm_srli_si128(tmp2, 10); //c:0,3,6,9,12,15, 0,0,0,0,0,0,0,0,0,0,
                    tmp3 = _mm_slli_si128(tmp3, 10); //0,0,0,0,0,0,0,0,0,0, c:0,3,6,9,12,15,
                    tmp4 = _mm_srli_si128(tmp1, 11); //b:1,4,7,10,13,0,0,0,0,0,0,0,0,0,0,0
                    tmp4 = _mm_slli_si128(tmp4, 5); //0,0,0,0,0,b:1,4,7,10,13, 0,0,0,0,0,0
                    tmp4 = _mm_or_si128(tmp4, tmp3); //0,0,0,0,0,b:1,4,7,10,13,c:0,3,6,9,12,15,
                    tmp0 = _mm_srli_si128(tmp0, 11); //a:2,5,8,11,14, 0,0,0,0,0,0,0,0,0,0,0,
                    c = from_vec<T>(_mm_or_si128(tmp4, tmp0)); //a:2,5,8,11,14,b:1,4,7,10,13,c:0,3,6,9,12,15,
                    return;
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    /*auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));*/
                    /*auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);*/
                    /*auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);*/
                    /*return;*/
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    /*auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));*/
                    /*auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);*/
                    /*auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);*/
                    /*return;*/
                }
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    /*auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));*/
                    /*auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);*/
                    /*auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);*/
                    /*return;*/
                }
            }
            strided_load(data, a.lo, b.lo, c.lo);
            strided_load(data + N / 2 * 3, a.hi, b.hi, c.hi);
        }
    }
} // namespace ui::x86

#endif // AMT_ARCH_X86_LOAD_HPP

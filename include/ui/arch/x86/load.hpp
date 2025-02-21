#ifndef AMT_ARCH_X86_LOAD_HPP
#define AMT_ARCH_X86_LOAD_HPP

#include "cast.hpp"
#include <algorithm>
#include <utility>

namespace ui::x86 {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto load(T val) noexcept -> Vec<N, T> {
        static constexpr auto size = N * sizeof(T);
        if constexpr (N == 1) {
            return { .val = static_cast<T>(val) };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (size == sizeof(__m128)) {
                    return from_vec<T>(_mm_set1_ps(val));
                }

                if constexpr (size * 2 == sizeof(__m128)) {
                    return load<2 * N>(val).lo;
                }

                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                if constexpr (size == sizeof(__m256)) {
                    return from_vec<T>(_mm256_set1_ps(val));
                }

                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (size == sizeof(__m512)) {
                    return from_vec<T>(_mm512_set1_ps(val));
                }
                #endif
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (size == sizeof(__m128)) {
                    return from_vec<T>(_mm_set1_pd(val));
                }

                if constexpr (size * 2 == sizeof(__m128)) {
                    return load<2 * N>(val).lo;
                }

                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                if constexpr (size == sizeof(__m256)) {
                    return from_vec<T>(_mm256_set1_pd(val));
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (size == sizeof(__m512)) {
                    return from_vec<T>(_mm512_set1_pd(val));
                }
                #endif
            } else {
                auto temp = static_cast<std::make_signed_t<T>>(val);
                if constexpr (size == sizeof(__m128)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_set1_epi8(temp));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_set1_epi16(temp));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_set1_epi32(temp));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_set1_epi64x((temp)));
                    }
                }

                if constexpr (size * 2 == sizeof(__m128)) {
                    return load<2 * N>(val).lo;
                }

                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                if constexpr (size == sizeof(__m256)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_set1_epi8(temp));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_set1_epi16(temp));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_set1_epi32(temp));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_set1_epi64x(temp));
                    }
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (size == sizeof(__m512)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_set1_epi8(temp));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_set1_epi16(temp));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_set1_epi32(temp));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_set1_epi64(temp));
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

    // https://android.googlesource.com/platform/external/neon_2_sse/+/48fc208e1a0026f2b9a64638eaaa83f1bc8408aa/NEON_2_SSE.h
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b
    ) noexcept {
        using namespace constants;
        static constexpr auto size = N * sizeof(T);
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
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto t0 = _mm_loadu_ps(data); // [d0, d1, d2, d3]
                    auto temp = from_vec<T>(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(t0), 0b11'10'01'00)));
                    a = temp.lo;
                    b = temp.hi;
                    return;
                }
            } else if constexpr (sizeof(T) == 1) {
                if constexpr (size == sizeof(__m128)) {
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
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto even_mask = *reinterpret_cast<__m128i const*>(mask8_16_even_odd);
                    auto temp = from_vec<T>(_mm_shuffle_epi8(t0, even_mask));
                    a = temp.lo;
                    b = temp.hi;
                    return;
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (size == sizeof(__m128)) {
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
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto even_mask = *reinterpret_cast<__m128i const*>(mask8_32_even_odd);
                    auto temp = from_vec<T>(_mm_shuffle_epi8(t0, even_mask));
                    a = temp.lo;
                    b = temp.hi;
                    return;
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (size == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto tmp0 = _mm_shuffle_epi32 (t0, 216);
                    auto tmp1 = _mm_shuffle_epi32 (t1, 216);
                    a = from_vec<T>(_mm_unpacklo_epi64(tmp0, tmp1));
                    b = from_vec<T>(_mm_unpackhi_epi64(tmp0, tmp1));
                    return;
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto temp = from_vec<T>(_mm_shuffle_epi32(t0, 0b11'01'10'00));
                    a = temp.lo;
                    b = temp.hi;
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
        static constexpr auto size = N * sizeof(T);
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
            } else if constexpr (sizeof(T) == 1) {
                if constexpr (size == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);

                    alignas(16) static constexpr std::int8_t mask8_0[16] = {0,3,6,9,12,15,1,4,7,10,13,2,5,8,11,14};
                    alignas(16) static constexpr std::int8_t mask8_1[16] = {2,5,8,11,14,0,3,6,9,12,15,1,4,7,10,13};
                    alignas(16) static constexpr std::int8_t mask8_2[16] = {1,4,7,10,13,2,5,8,11,14,0,3,6,9,12,15};

                    auto tmp0 = _mm_shuffle_epi8(t0, *reinterpret_cast<__m128i const*>(mask8_0));
                    auto tmp1 = _mm_shuffle_epi8(t1, *reinterpret_cast<__m128i const*>(mask8_1));
                    auto tmp2 = _mm_shuffle_epi8(t2, *reinterpret_cast<__m128i const*>(mask8_2));

                    auto tmp3 = _mm_slli_si128(tmp0, 10);   //0,0,0,0,0,0,0,0,0,0,a0,a3,a6,a9,a12,a15
                    tmp3 = _mm_alignr_epi8(tmp1, tmp3, 10); //a:0,3,6,9,12,15,b:2,5,8,11,14,x,x,x,x,x
                    tmp3 = _mm_slli_si128(tmp3, 5);         //0,0,0,0,0,a:0,3,6,9,12,15,b:2,5,8,11,14,
                    tmp3 = _mm_srli_si128(tmp3, 5);         //a:0,3,6,9,12,15,b:2,5,8,11,14,:0,0,0,0,0

                    auto tmp4 = _mm_slli_si128(tmp2, 11);   //0,0,0,0,0,0,0,0,0,0,0,0, 1,4,7,10,13,
                    a = from_vec<T>(_mm_or_si128(tmp4, tmp3)); //a:0,3,6,9,12,15,b:2,5,8,11,14,c:1,4,7,10,13,

                    tmp3 = _mm_slli_si128(tmp0, 5);     //0,0,0,0,0,a:0,3,6,9,12,15,1,4,7,10,13,
                    tmp3 = _mm_srli_si128(tmp3, 11);    //a:1,4,7,10,13, 0,0,0,0,0,0,0,0,0,0,0
                    tmp4 = _mm_srli_si128(tmp1, 5);     //b:0,3,6,9,12,15,C:1,4,7,10,13, 0,0,0,0,0
                    tmp4 = _mm_slli_si128(tmp4, 5);     //0,0,0,0,0,b:0,3,6,9,12,15,C:1,4,7,10,13,
                    tmp4 = _mm_or_si128(tmp4, tmp3);    //a:1,4,7,10,13,b:0,3,6,9,12,15,C:1,4,7,10,13,
                    tmp4 = _mm_slli_si128(tmp4, 5);     //0,0,0,0,0,a:1,4,7,10,13,b:0,3,6,9,12,15,
                    tmp4 = _mm_srli_si128(tmp4, 5);     //a:1,4,7,10,13,b:0,3,6,9,12,15,0,0,0,0,0

                    tmp3 = _mm_srli_si128(tmp2, 5);     //c:2,5,8,11,14,0,3,6,9,12,15,0,0,0,0,0
                    tmp3 = _mm_slli_si128(tmp3, 11);    //0,0,0,0,0,0,0,0,0,0,0,c:2,5,8,11,14,
                    b = from_vec<T>(_mm_or_si128(tmp4, tmp3)); //a:1,4,7,10,13,b:0,3,6,9,12,15,c:2,5,8,11,14,

                    tmp3 = _mm_srli_si128(tmp2, 10);    //c:0,3,6,9,12,15, 0,0,0,0,0,0,0,0,0,0,
                    tmp3 = _mm_slli_si128(tmp3, 10);    //0,0,0,0,0,0,0,0,0,0, c:0,3,6,9,12,15,
                    tmp4 = _mm_srli_si128(tmp1, 11);    //b:1,4,7,10,13,0,0,0,0,0,0,0,0,0,0,0
                    tmp4 = _mm_slli_si128(tmp4, 5);     //0,0,0,0,0,b:1,4,7,10,13, 0,0,0,0,0,0
                    tmp4 = _mm_or_si128(tmp4, tmp3);    //0,0,0,0,0,b:1,4,7,10,13,c:0,3,6,9,12,15,
                    tmp0 = _mm_srli_si128(tmp0, 11);    //a:2,5,8,11,14, 0,0,0,0,0,0,0,0,0,0,0,
                    c = from_vec<T>(_mm_or_si128(tmp4, tmp0)); //a:2,5,8,11,14,b:1,4,7,10,13,c:0,3,6,9,12,15,
                    return;
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t2 = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(data) + 1);
                    alignas(16) static constexpr std::int8_t mask8_0[16] = {0,3,6,9,12,15, 1,4,7,10,13, 2,5,8,11,14};
                    alignas(16) static constexpr std::int8_t mask8_1[16] = {2,5, 0,3,6, 1,4,7, 0,0,0,0,0,0,0,0};

                    auto tmp0 = _mm_shuffle_epi8(t0, *reinterpret_cast<__m128i const*>(mask8_0));
                    auto tmp1 = _mm_shuffle_epi8(t2, *reinterpret_cast<__m128i const*>(mask8_1));

                    t0 = _mm_slli_si128(tmp0, 10);
                    t0 = _mm_srli_si128(t0, 10); //a0,a3,a6,b1,b4,b7, 0,0,0,0,0,0,0,0,0,0
                    t2 = _mm_slli_si128(tmp1, 6); //0,0,0,0,0,0,c2,c5,x,x,x,x,x,x,x,x
                    t0 = _mm_or_si128(t0, t2); //a0,a3,a6,b1,b4,b7,c2,c5 x,x,x,x,x,x,x,x
                    a = from_vec<T>(t0).lo;

                    auto t1 = _mm_slli_si128(tmp0, 5); //0,0,0,0,0,0,0,0,0,0,0, a1,a4,a7,b2,b5,
                    t1 = _mm_srli_si128(t1, 11); //a1,a4,a7,b2,b5,0,0,0,0,0,0,0,0,0,0,0,
                    t2 = _mm_srli_si128(tmp1, 2); //c0,c3,c6,c1,c4,c7,x,x,x,x,x,x,x,x,0,0
                    t2 = _mm_slli_si128(t2, 5); //0,0,0,0,0,c0,c3,c6,0,0,0,0,0,0,0,0
                    t1 = _mm_or_si128(t1, t2); //a1,a4,a7,b2,b5,c0,c3,c6,x,x,x,x,x,x,x,x
                    b = from_vec<T>(t1).lo;

                    tmp0 = _mm_srli_si128(tmp0, 11); //a2,a5,b0,b3,b6,0,0,0,0,0,0,0,0,0,0,0,
                    t2 = _mm_srli_si128(tmp1, 5); //c1,c4,c7,0,0,0,0,0,0,0,0,0,0,0,0,0
                    t2 = _mm_slli_si128(t2, 5); //0,0,0,0,0,c1,c4,c7,
                    t2 = _mm_or_si128(tmp0, t2); //a2,a5,b0,b3,b6,c1,c4,c7,x,x,x,x,x,x,x,x
                    c = from_vec<T>(t2).lo;
                }

            } else if constexpr (sizeof(T) == 2) {
                if constexpr (size == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);

                    alignas(16) static constexpr std::int8_t mask16_0[16] = {0,1, 6,7, 12,13, 2,3, 8,9, 14,15, 4,5, 10,11};
                    alignas(16) static constexpr std::int8_t mask16_1[16] = {2,3, 8,9, 14,15, 4,5, 10,11, 0,1, 6,7, 12,13};
                    alignas(16) static constexpr std::int8_t mask16_2[16] = {4,5, 10,11, 0,1, 6,7, 12,13, 2,3, 8,9, 14,15};

                    auto tmp0 = _mm_shuffle_epi8(t0, *reinterpret_cast<__m128i const*>(mask16_0));
                    auto tmp1 = _mm_shuffle_epi8(t1, *reinterpret_cast<__m128i const*>(mask16_1));
                    auto tmp2 = _mm_shuffle_epi8(t2, *reinterpret_cast<__m128i const*>(mask16_2));

                    auto tmp3 = _mm_slli_si128(tmp0, 10);       //0,0,0,0,0,a0,a3,a6,
                    tmp3      = _mm_alignr_epi8(tmp1,tmp3, 10); //a0,a3,a6,b1,b4,b7,x,x
                    tmp3      = _mm_slli_si128(tmp3, 4);        //0,0, a0,a3,a6,b1,b4,b7
                    tmp3      = _mm_srli_si128(tmp3, 4);        //a0,a3,a6,b1,b4,b7,0,0
                    auto tmp4 = _mm_slli_si128(tmp2, 12);       //0,0,0,0,0,0, c2,c5,
                    a         = from_vec<T>(_mm_or_si128(tmp4,tmp3));   //a0,a3,a6,b1,b4,b7,c2,c5

                    tmp3 = _mm_slli_si128(tmp0, 4);     //0,0,a0,a3,a6,a1,a4,a7
                    tmp3 = _mm_srli_si128(tmp3, 10);    //a1,a4,a7, 0,0,0,0,0
                    tmp4 = _mm_srli_si128(tmp1, 6);     //b2,b5,b0,b3,b6,0,0
                    tmp4 = _mm_slli_si128(tmp4, 6);     //0,0,0,b2,b5,b0,b3,b6,
                    tmp4 = _mm_or_si128(tmp4, tmp3);    //a1,a4,a7,b2,b5,b0,b3,b6,
                    tmp4 =  _mm_slli_si128(tmp4,6);     //0,0,0,a1,a4,a7,b2,b5,
                    tmp4 = _mm_srli_si128(tmp4, 6);     //a1,a4,a7,b2,b5,0,0,0,
                    tmp3 = _mm_srli_si128(tmp2, 4);     //c0,c3,c6, c1,c4,c7,0,0
                    tmp3 = _mm_slli_si128(tmp3, 10);    //0,0,0,0,0,c0,c3,c6,
                    b    = from_vec<T>(_mm_or_si128(tmp4, tmp3)); //a1,a4,a7,b2,b5,c0,c3,c6,

                    tmp3 = _mm_srli_si128(tmp2, 10);     //c1,c4,c7, 0,0,0,0,0
                    tmp3 = _mm_slli_si128(tmp3, 10);     //0,0,0,0,0, c1,c4,c7,
                    tmp4 = _mm_srli_si128(tmp1, 10);     //b0,b3,b6,0,0, 0,0,0
                    tmp4 = _mm_slli_si128(tmp4, 4);      //0,0, b0,b3,b6,0,0,0
                    tmp4 = _mm_or_si128(tmp4, tmp3);     //0,0, b0,b3,b6,c1,c4,c7,
                    tmp0 = _mm_srli_si128(tmp0, 12);     //a2,a5,0,0,0,0,0,0
                    c    = from_vec<T>(_mm_or_si128(tmp4, tmp0)); //a2,a5,b0,b3,b6,c1,c4,c7,
                    return;
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto val0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto val2 = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(data) + 1);

                    alignas(16) static constexpr std::int8_t mask16[16] = {0,1, 6,7, 12,13, 2,3, 8,9, 14,15, 4,5, 10,11};

                    auto tmp0 = _mm_shuffle_epi8(val0, *reinterpret_cast<__m128i const*>(mask16)); //a0, a3, b2,a1, b0, b3, a2, b1
                    auto tmp1 = _mm_shufflelo_epi16(val2, 201); //11 00 10 01     : c1, c2, c0, c3,
                    val0 = _mm_slli_si128(tmp0, 10);
                    val0 = _mm_srli_si128(val0, 10); //a0, a3, b2, 0,0, 0,0,
                    val2 = _mm_slli_si128(tmp1, 14); //0,0,0,0,0,0,0,c1
                    val2 = _mm_srli_si128(val2, 8); //0,0,0,c1,0,0,0,0
                    val0 = _mm_or_si128(val0, val2); //a0, a3, b2, c1, x,x,x,x
                    a = from_vec<T>(val0).lo;

                    auto val1 = _mm_slli_si128(tmp0, 4); //0,0,0,0,0,a1, b0, b3
                    val1 = _mm_srli_si128(val1, 10); //a1, b0, b3, 0,0, 0,0,
                    val2 = _mm_srli_si128(tmp1, 2); //c2, 0,0,0,0,0,0,0,
                    val2 = _mm_slli_si128(val2, 6); //0,0,0,c2,0,0,0,0
                    val1 = _mm_or_si128(val1, val2); //a1, b0, b3, c2, x,x,x,x
                    b = from_vec<T>(val1).lo;

                    tmp0 = _mm_srli_si128(tmp0, 12); //a2, b1,0,0,0,0,0,0
                    tmp1 = _mm_srli_si128(tmp1, 4);
                    tmp1 = _mm_slli_si128(tmp1, 4); //0,0,c0, c3,
                    val2 = _mm_or_si128(tmp0, tmp1); //a2, b1, c0, c3,
                    c = from_vec<T>(val2).lo;
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (size == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);

                    auto tmp0 = _mm_shuffle_epi32(
                        t0,
                        0b10'01'11'00 // [2, 1, 3, 0]
                    ); //a0,a3,a1,a2
                    auto tmp1 = _mm_shuffle_epi32(t1, swap_hi_low32); //b2,b3,b0,b1
                    auto tmp2 = _mm_shuffle_epi32(
                        t2,
                        0b11'00'10'01 // [3, 0, 2, 1]
                    ); //c1,c2, c0,c3

                    auto tmp3 = _mm_unpacklo_epi32(tmp1, tmp2); //b2,c1, b3,c2
                    a = from_vec<T>(_mm_unpacklo_epi64(tmp0, tmp3)); //a0,a3,b2,c1
                    tmp0 = _mm_unpackhi_epi32(tmp0, tmp1); //a1,b0, a2,b1

                    auto tmp4 = _mm_shuffle_epi32(tmp0, swap_hi_low32); //a2,b1, a1,b0,
                    b = from_vec<T>(_mm_unpackhi_epi64(tmp4, tmp3)); //a1,b0, b3,c2

                    c = from_vec<T>(_mm_unpackhi_epi64(tmp0, tmp2)); //a2,b1, c0,c3
                    return;
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto val0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto val2 = _mm_loadl_epi64(reinterpret_cast<__m128i const*>(data) + 1);
                    val0 = _mm_shuffle_epi32(
                        val0,
                        0b10'01'11'00 // [2, 1, 3, 0]
                    ); //a0,b1, a1, b0
                    a = from_vec<T>(val0).lo;

                    val2 =  _mm_slli_si128(val2, 8); //x, x,c0,c1,
                    auto val1 =  _mm_unpackhi_epi32(val0, val2); //a1,c0, b0, c1
                    b = from_vec<T>(val1).lo;

                    val2 =  _mm_srli_si128(val1, 8); //b0, c1, x, x,
                    c = from_vec<T>(val2).lo;
                }
            }
            strided_load(data, a.lo, b.lo, c.lo);
            strided_load(data + N / 2 * 3, a.hi, b.hi, c.hi);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c,
        Vec<N, T>& UI_RESTRICT d
    ) noexcept {
        static constexpr auto size = N * sizeof(T);
        using namespace constants;
        if constexpr (N == 1) {
            a.val = data[0];
            b.val = data[1];
            c.val = data[2];
            d.val = data[3];
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N * sizeof(T) == sizeof(__m128)) {
                    auto t0 = _mm_loadu_ps(data); // [d0, d1, d2, d3]
                    auto t1 = _mm_loadu_ps(data + 4); // [d4, d5, d6, d7]
                    auto t2 = _mm_loadu_ps(data + 8);
                    auto t3 = _mm_loadu_ps(data + 16);
                    auto tmp0 = _mm_unpacklo_ps(t0, t1);
                    auto tmp2 = _mm_unpacklo_ps(t2, t3);
                    auto tmp1 = _mm_unpackhi_ps(t0, t1);
                    auto tmp3 = _mm_unpackhi_ps(t2, t3);
                    a = from_vec<T>(_mm_movelh_ps(tmp0, tmp2));
                    b = from_vec<T>(_mm_movehl_ps(tmp2, tmp0));
                    c = from_vec<T>(_mm_movelh_ps(tmp1, tmp3));
                    d = from_vec<T>(_mm_movehl_ps(tmp3, tmp1));
                    return;
                }
            } else if constexpr (sizeof(T) == 1) {
                if constexpr (size == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);
                    auto t3 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 3);
                    auto tmp0 = _mm_unpacklo_epi8(t0,t1); //a0,b0, a1,b1, a2,b2, a3,b3,....a7,b7
                    auto tmp1 = _mm_unpacklo_epi8(t2,t3); //c0,d0, c1,d1, c2,d2, c3,d3,... c7,d7
                    auto tmp2 = _mm_unpackhi_epi8(t0,t1); //a8,b8, a9,b9, a10,b10, a11,b11,...a15,b15
                    auto tmp3 = _mm_unpackhi_epi8(t2,t3); //c8,d8, c9,d9, c10,d10, c11,d11,...c15,d15

                    t0 = _mm_unpacklo_epi8(tmp0, tmp2); //a0,a8, b0,b8,  a1,a9, b1,b9, ....a3,a11, b3,b11
                    t1 = _mm_unpackhi_epi8(tmp0, tmp2); //a4,a12, b4,b12, a5,a13, b5,b13,....a7,a15,b7,b15
                    t2 = _mm_unpacklo_epi8(tmp1, tmp3); //c0,c8, d0,d8, c1,c9, d1,d9.....d3,d11
                    t3 = _mm_unpackhi_epi8(tmp1, tmp3); //c4,c12,d4,d12, c5,c13, d5,d13,....d7,d15

                    tmp0 =  _mm_unpacklo_epi32(t0, t2); ///a0,a8, b0,b8, c0,c8,  d0,d8, a1,a9, b1,b9, c1,c9, d1,d9
                    tmp1 =  _mm_unpackhi_epi32(t0, t2); //a2,a10, b2,b10, c2,c10, d2,d10, a3,a11, b3,b11, c3,c11, d3,d11
                    tmp2 =  _mm_unpacklo_epi32(t1, t3); //a4,a12, b4,b12, c4,c12, d4,d12, a5,a13, b5,b13, c5,c13, d5,d13,
                    tmp3 =  _mm_unpackhi_epi32(t1, t3); //a6,a14, b6,b14, c6,c14, d6,d14, a7,a15,b7,b15,c7,c15,d7,d15

                    a = from_vec<T>(_mm_unpacklo_epi8(tmp0, tmp2)); //a0,a4,a8,a12,b0,b4,b8,b12,c0,c4,c8,c12,d0,d4,d8,d12
                    b = from_vec<T>(_mm_unpackhi_epi8(tmp0, tmp2)); //a1,a5, a9, a13, b1,b5, b9,b13, c1,c5, c9, c13, d1,d5, d9,d13
                    c = from_vec<T>(_mm_unpacklo_epi8(tmp1, tmp3)); //a2,a6, a10,a14, b2,b6, b10,b14,c2,c6, c10,c14, d2,d6, d10,d14
                    d = from_vec<T>(_mm_unpackhi_epi8(tmp1, tmp3)); //a3,a7, a11,a15, b3,b7, b11,b15,c3,c7, c11, c15,d3,d7, d11,d15
                    return;
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    alignas(16) static constexpr std::int8_t mask4_8[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

                    auto val0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto val1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);

                    auto sh0 = _mm_shuffle_epi8(val0, *reinterpret_cast<__m128i const*>(mask4_8));
                    auto sh1 = _mm_shuffle_epi8(val1, *reinterpret_cast<__m128i const*>(mask4_8));

                    val0 = _mm_unpacklo_epi32(sh0, sh1); //0,4,8,12,16,20,24,28, 1,5,9,13,17,21,25,29
                    auto t0 = from_vec<T>(val0);
                    a = t0.lo;
                    b = t0.hi;

                    auto val2 = _mm_unpackhi_epi32(sh0, sh1);

                    t0 = from_vec<T>(val2);
                    c = t0.lo;
                    d = t0.hi;
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (size == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);
                    auto t3 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 3);

                    t0 = _mm_unpacklo_epi16(t0,t1); //a0,b0, a1,b1, a2,b2, a3,b3,
                    t1 = _mm_unpacklo_epi16(t2,t3); //c0,d0, c1,d1, c2,d2, c3,d3,
                    t2 = _mm_unpackhi_epi16(t0,t1); //a4,b4, a5,b5, a6,b6, a7,b7
                    t3 = _mm_unpackhi_epi16(t2,t3); //c4,d4, c5,d5, c6,d6, c7,d7
                    auto tmp0 = _mm_unpacklo_epi16(t0, t2); //a0,a4, b0,b4, a1,a5, b1,b5
                    auto tmp1 = _mm_unpackhi_epi16(t0, t2); //a2,a6, b2,b6, a3,a7, b3,b7
                    auto tmp2 = _mm_unpacklo_epi16(t1, t3); //c0,c4, d0,d4, c1,c5, d1,d5
                    auto tmp3 = _mm_unpackhi_epi16(t1, t3); //c2,c6, d2,d6, c3,c7, d3,d7
                    a = from_vec<T>(_mm_unpacklo_epi64(tmp0, tmp2)); //a0,a4, b0,b4, c0,c4, d0,d4,
                    b = from_vec<T>(_mm_unpackhi_epi64(tmp0, tmp2)); //a1,a5, b1,b5, c1,c5, d1,d5
                    c = from_vec<T>(_mm_unpacklo_epi64(tmp1, tmp3)); //a2,a6, b2,b6, c2,c6, d2,d6,
                    d = from_vec<T>(_mm_unpackhi_epi64(tmp1, tmp3)); //a3,a7, b3,b7, c3,c7, d3,d7
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    alignas(16) static constexpr std::int8_t mask4_8[16] = {0,1, 8,9, 2,3, 10,11, 4,5, 12,13, 6,7, 14,15};

                    auto val0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto val1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);

                    auto sh0 = _mm_shuffle_epi8(val0, *reinterpret_cast<__m128i const*>(mask4_8));
                    auto sh1 = _mm_shuffle_epi8(val1, *reinterpret_cast<__m128i const*>(mask4_8));

                    val0 = _mm_unpacklo_epi32(sh0, sh1); //0,4,8,12,16,20,24,28, 1,5,9,13,17,21,25,29
                    auto t0 = from_vec<T>(val0);
                    a = t0.lo;
                    b = t0.hi;

                    auto val2 = _mm_unpackhi_epi32(sh0, sh1);

                    t0 = from_vec<T>(val2);
                    c = t0.lo;
                    d = t0.hi;
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (size == sizeof(__m128)) {
                    auto t0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto t1 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);
                    auto t2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 2);
                    auto t3 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 3);

                    auto tmp0 = _mm_unpacklo_epi32(t0,t1);
                    auto tmp1 = _mm_unpacklo_epi32(t2,t3);
                    auto tmp2 = _mm_unpackhi_epi32(t0,t1);
                    auto tmp3 = _mm_unpackhi_epi32(t2,t3);
                    a = from_vec<T>(_mm_unpacklo_epi64(tmp0, tmp1));
                    b = from_vec<T>(_mm_unpackhi_epi64(tmp0, tmp1));
                    c = from_vec<T>(_mm_unpacklo_epi64(tmp2, tmp3));
                    d = from_vec<T>(_mm_unpackhi_epi64(tmp2, tmp3));
                } else if constexpr (size * 2 == sizeof(__m128)) {
                    auto val0 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data));
                    auto val2 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(data) + 1);

                    auto t0 = from_vec<T>(_mm_unpacklo_epi32(val0, val2)); //a0, c0, a1,c1,
                    auto t1 = from_vec<T>(_mm_unpackhi_epi32(val0, val2)); //b0,d0, b1, d1
                    a = t0.lo;
                    b = t0.hi;
                    c = t1.lo;
                    d = t1.hi;
                }
            }
            strided_load(data, a.lo, b.lo, c.lo, d.lo);
            strided_load(data + N / 2 * 4, a.hi, b.hi, c.hi, d.hi);
        }
    }
} // namespace ui::x86

#endif // AMT_ARCH_X86_LOAD_HPP

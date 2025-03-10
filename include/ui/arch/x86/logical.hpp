#ifndef AMT_UI_ARCH_X86_LOGICAL_HPP
#define AMT_UI_ARCH_X86_LOGICAL_HPP

#include "cast.hpp"
#include "../emul/logical.hpp"
#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::x86 {

// MARK: Bitwise And
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_and(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_and(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                return from_vec<T>(_mm_and_si128(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (size * 2 == sizeof(__m128)) {
                return bitwise_and(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                return from_vec<T>(_mm256_and_si256(to_vec(lhs), to_vec(rhs)));
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                return from_vec<T>(_mm512_and_si512(to_vec(lhs), to_vec(rhs)));
            }
            #endif

            return join(
                bitwise_and<false>(lhs.lo, rhs.lo),
                bitwise_and<false>(lhs.hi, rhs.hi)
            );
        }
    }

// !MARK

// MARK: Bitwise XOR
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_xor(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_xor(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                return from_vec<T>(_mm_xor_si128(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return bitwise_xor(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                return from_vec<T>(_mm256_xor_si256(to_vec(lhs), to_vec(rhs)));
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                return from_vec<T>(_mm512_xor_si512(to_vec(lhs), to_vec(rhs)));
            }
            #endif

            return join(
                bitwise_xor<false>(lhs.lo, rhs.lo),
                bitwise_xor<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Negation
    template <bool Merge = true, std::size_t N, typename T>
        requires (std::is_floating_point_v<T> || std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto negate(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(v);
        if constexpr (N == 1) {
            return emul::negate(v);
        } else {
            if constexpr (std::same_as<T, float>) {
                static constexpr std::uint32_t sign_mask = 0x8000'0000;
                auto mask = Vec<N, std::uint32_t>::load(sign_mask);
                return rcast<T>(bitwise_xor(rcast<std::uint32_t>(v), mask));
            } else if constexpr (std::same_as<T, double>) {
                static constexpr std::uint64_t sign_mask = 0x8000'0000'0000'0000;
                auto mask = Vec<N, std::uint64_t>::load(sign_mask);
                return rcast<T>(bitwise_xor(rcast<std::uint64_t>(v), mask));
            } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                static constexpr std::uint16_t sign_mask = 0x8000;
                auto mask = Vec<N, std::uint16_t>::load(sign_mask);
                return rcast<T>(bitwise_xor(rcast<std::uint16_t>(v), mask));
            } else {
                if constexpr (size == sizeof(__m128)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_sub_epi8(_mm_setzero_si128(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_sub_epi16(_mm_setzero_si128(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_sub_epi32(_mm_setzero_si128(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_sub_epi64(_mm_setzero_si128(), to_vec(v)));
                    }
                } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                    return negate(from_vec<T>(fit_to_vec(v))).lo;
                }

                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (size == sizeof(__m256)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_sub_epi8(_mm256_setzero_si256(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_sub_epi16(_mm256_setzero_si256(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_sub_epi32(_mm256_setzero_si256(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_sub_epi64(_mm256_setzero_si256(), to_vec(v)));
                    }
                }
                #endif

                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (size == sizeof(__m512)) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_sub_epi8(_mm512_setzero_si512(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_sub_epi16(_mm512_setzero_si512(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_sub_epi32(_mm512_setzero_si512(), to_vec(v)));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_sub_epi64(_mm512_setzero_si512(), to_vec(v)));
                    }
                }
                #endif
                return join(
                    negate<false>(v.lo),
                    negate<false>(v.hi)
                );
            }
        }
    }

    template <bool Merge, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <std::size_t N, std::integral T>
        requires (std::is_arithmetic_v<T> && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_negate(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::sat_negate(v);
        } else {
            return sat_sub(Vec<N, T>{}, v);
        }
    }
// !MARK

// MARK: Bitwise Not
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_not(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(v);
        if constexpr (N == 1) {
            return emul::bitwise_not(v);
        } else {
            if constexpr (size == sizeof(__m128)) {
                if constexpr (sizeof(T) == 1) {
                    auto ci = _mm_cmpeq_epi8(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm_andnot_si128(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 2) {
                    auto ci = _mm_cmpeq_epi16(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm_andnot_si128(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 4) {
                    auto ci = _mm_cmpeq_epi32(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm_andnot_si128(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 8) {
                    auto ci = _mm_cmpeq_epi64(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm_andnot_si128(to_vec(v), ci));
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return bitwise_not(from_vec<T>(fit_to_vec(v))).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                if constexpr (sizeof(T) == 1) {
                    auto ci = _mm256_cmpeq_epi8(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm256_andnot_si256(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 2) {
                    auto ci = _mm256_cmpeq_epi16(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm256_andnot_si256(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 4) {
                    auto ci = _mm256_cmpeq_epi32(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm256_andnot_si256(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 8) {
                    auto ci = _mm256_cmpeq_epi64(to_vec(v), to_vec(v));
                    return from_vec<T>(_mm256_andnot_si256(to_vec(v), ci));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                if constexpr (sizeof(T) == 1) {
                    auto ci = _mm512_set1_epi8(0xFF);
                    return from_vec<T>(_mm512_xor_si512(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 2) {
                    auto ci = _mm512_set1_epi8(0xFFFF);
                    return from_vec<T>(_mm512_xor_si512(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 4) {
                    auto ci = _mm512_set1_epi8(0xFFFF'FFFF);
                    return from_vec<T>(_mm512_xor_si512(to_vec(v), ci));
                } else if constexpr (sizeof(T) == 8) {
                    auto ci = _mm512_set1_epi8(0xFFFF'FFFF'FFFF'FFFFf);
                    return from_vec<T>(_mm512_xor_si512(to_vec(v), ci));
                }
            }
            #endif
            return join(
                bitwise_not<false>(v.lo),
                bitwise_not<false>(v.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise OR
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_or(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_or(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                return from_vec<T>(_mm_or_si128(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return bitwise_or(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                return from_vec<T>(_mm256_or_si256(to_vec(lhs), to_vec(rhs)));
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                return from_vec<T>(_mm512_or_si512(to_vec(lhs), to_vec(rhs)));
            }
            #endif
            return join(
                bitwise_or<false>(lhs.lo, rhs.lo),
                bitwise_or<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise Not-And ~lhs & rhs
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_notand(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_notand(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                return from_vec<T>(_mm_andnot_si128(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return bitwise_notand(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (size == sizeof(__m256)) {
                return from_vec<T>(_mm256_andnot_si256(to_vec(lhs), to_vec(rhs)));
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                return from_vec<T>(_mm512_andnot_si512(to_vec(lhs), to_vec(rhs)));
            }
            #endif
            return join(
                bitwise_notand<false>(lhs.lo, rhs.lo),
                bitwise_notand<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise Or-Not lhs | ~rhs
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_ornot(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_ornot(lhs, rhs);
        } else {
            return bitwise_not(bitwise_notand(lhs, rhs));
        }
    }
// !MARK
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_LOGICAL_HPP

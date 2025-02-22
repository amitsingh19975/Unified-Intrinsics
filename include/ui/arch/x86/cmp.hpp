#ifndef AMT_UI_ARCH_X86_CMP_HPP
#define AMT_UI_ARCH_X86_CMP_HPP

#include "cast.hpp"
#include "../emul/cmp.hpp"
#include "logical.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::x86 {

// MARK: Bitwise equal and 'and' test
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::equal_t op
    ) noexcept -> mask_t<N, T> {
        using result_t = mask_inner_t<T>;
        static constexpr auto size = sizeof(lhs);

        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            if constexpr (size == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm_castps_si128(_mm_cmpeq_ps(to_vec(lhs), to_vec(rhs))));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm_castpd_si128(_mm_cmpeq_pd(to_vec(lhs), to_vec(rhs))));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (sizeof(T) == 1) {
                    return from_vec<result_t>(_mm_cmpeq_epi8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    return from_vec<result_t>(_mm_cmpeq_epi16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    return from_vec<result_t>(_mm_cmpeq_epi32(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<result_t>(_mm_cmpeq_epi64(to_vec(lhs), to_vec(rhs)));
                }
            } else if constexpr (size * 2 == sizeof(__m128)) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm256_castps_si256(_mm256_cmp_ps(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm256_castpd_si256(_mm256_cmp_pd(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (sizeof(T) == 1) {
                    return from_vec<result_t>(_mm256_cmp_epi8(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                } else if constexpr (sizeof(T) == 2) {
                    return from_vec<result_t>(_mm256_cmp_epi16(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                } else if constexpr (sizeof(T) == 4) {
                    return from_vec<result_t>(_mm256_cmp_epi32(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<result_t>(_mm256_cmp_epi64(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                }
            } else if constexpr (size * 2 == sizeof(__m256)) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<result_t>(_mm512_castps_si512(_mm512_cmp_ps(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<result_t>(_mm512_castpd_si512(_mm512_cmp_pd(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));
                } else if constexpr (sizeof(T) == 1) {
                    return from_vec<result_t>(_mm512_cmp_epi8(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                } else if constexpr (sizeof(T) == 2) {
                    return from_vec<result_t>(_mm512_cmp_epi16(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                } else if constexpr (sizeof(T) == 4) {
                    return from_vec<result_t>(_mm512_cmp_epi32(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                } else if constexpr (sizeof(T) == 8) {
                    return from_vec<result_t>(_mm512_cmp_epi64(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));
                }
            } else if constexpr (size * 2 == sizeof(__m512)) {
                return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;
            }
            #endif
            return join(
                cmp(lhs.lo, rhs.lo, op),
                cmp(lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::and_test_t op
    ) noexcept -> mask_t<N, T> {
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            auto zero = Vec<N, T>{};
            auto ones = cmp(zero, zero, op::equal_t{});
            auto res = bitwise_and(lhs, rhs);
            return bitwise_xor(
                cmp(res, zero, op::equal_t{}),
                ones
            );
        }
    }

// !MARK

// MARK:  Greater than or equal to
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto cmp(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::greater_equal_t op
    ) noexcept -> mask_t<N, T> {
        if constexpr (N == 1) {
            return emul::cmp(lhs, rhs, op);
        } else {
            /*if constexpr (size == sizeof(__m128)) {*/
            /*    if constexpr (std::same_as<T, float>) {*/
            /*        return from_vec<result_t>(_mm_castps_si128(_mm_cmpge_ps(to_vec(lhs), to_vec(rhs))));*/
            /*    } else if constexpr (std::same_as<T, double>) {*/
            /*        return from_vec<result_t>(_mm_castpd_si128(_mm_cmpge_pd(to_vec(lhs), to_vec(rhs))));*/
            /*    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {*/
            /*        return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));*/
            /*    } else if constexpr (sizeof(T) == 1) {*/
            /*        return from_vec<result_t>(_mm_cmpge_epi8(to_vec(lhs), to_vec(rhs)));*/
            /*    } else if constexpr (sizeof(T) == 2) {*/
            /*        return from_vec<result_t>(_mm_cmpge_epi16(to_vec(lhs), to_vec(rhs)));*/
            /*    } else if constexpr (sizeof(T) == 4) {*/
            /*        return from_vec<result_t>(_mm_cmpge_epi32(to_vec(lhs), to_vec(rhs)));*/
            /*    } else if constexpr (sizeof(T) == 8) {*/
            /*        return from_vec<result_t>(_mm_cmpge_epi64(to_vec(lhs), to_vec(rhs)));*/
            /*    }*/
            /*} else if constexpr (size * 2 == sizeof(__m128)) {*/
            /*    return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;*/
            /*}*/
            /**/
            /*#if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX*/
            /*if constexpr (size == sizeof(__m256)) {*/
            /*    if constexpr (std::same_as<T, float>) {*/
            /*        return from_vec<result_t>(_mm256_castps_si256(_mm256_cmp_ps(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));*/
            /*    } else if constexpr (std::same_as<T, double>) {*/
            /*        return from_vec<result_t>(_mm256_castpd_si256(_mm256_cmp_pd(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));*/
            /*    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {*/
            /*        return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));*/
            /*    } else if constexpr (sizeof(T) == 1) {*/
            /*        return from_vec<result_t>(_mm256_cmp_epi8(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    } else if constexpr (sizeof(T) == 2) {*/
            /*        return from_vec<result_t>(_mm256_cmp_epi16(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    } else if constexpr (sizeof(T) == 4) {*/
            /*        return from_vec<result_t>(_mm256_cmp_epi32(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    } else if constexpr (sizeof(T) == 8) {*/
            /*        return from_vec<result_t>(_mm256_cmp_epi64(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    }*/
            /*} else if constexpr (size * 2 == sizeof(__m256)) {*/
            /*    return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;*/
            /*}*/
            /*#endif*/
            /**/
            /*#if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX*/
            /*if constexpr (size == sizeof(__m512)) {*/
            /*    if constexpr (std::same_as<T, float>) {*/
            /*        return from_vec<result_t>(_mm512_castps_si512(_mm512_cmp_ps(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));*/
            /*    } else if constexpr (std::same_as<T, double>) {*/
            /*        return from_vec<result_t>(_mm512_castpd_si512(_mm512_cmp_pd(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ)));*/
            /*    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {*/
            /*        return cast<result_t>(cmp(cast<float>(lhs), cast<float>(rhs), op));*/
            /*    } else if constexpr (sizeof(T) == 1) {*/
            /*        return from_vec<result_t>(_mm512_cmp_epi8(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    } else if constexpr (sizeof(T) == 2) {*/
            /*        return from_vec<result_t>(_mm512_cmp_epi16(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    } else if constexpr (sizeof(T) == 4) {*/
            /*        return from_vec<result_t>(_mm512_cmp_epi32(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    } else if constexpr (sizeof(T) == 8) {*/
            /*        return from_vec<result_t>(_mm512_cmp_epi64(to_vec(lhs), to_vec(rhs), _CMP_EQ_OQ));*/
            /*    }*/
            /*} else if constexpr (size * 2 == sizeof(__m512)) {*/
            /*    return cmp(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs)), op).lo;*/
            /*}*/
            /*#endif*/
            /*return join(*/
            /*    cmp(lhs.lo, rhs.lo, op),*/
            /*    cmp(lhs.hi, rhs.hi, op)*/
            /*);*/
        }
    }
// !MARK
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_CMP_HPP

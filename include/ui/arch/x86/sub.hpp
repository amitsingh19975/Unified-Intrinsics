#ifndef AMT_UI_ARCH_X86_SUB_HPP
#define AMT_UI_ARCH_X86_SUB_HPP

#include "cast.hpp"
#include "../emul/sub.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::x86 {

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::sub(lhs, rhs);
        } else {
            if constexpr (size == sizeof(__m128)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_sub_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_sub_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(sub(cast<float>(l), cast<float>(r)));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm_sub_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm_sub_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_sub_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_sub_epi64(l, r));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m128) && Merge) {
                return sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (size == sizeof(__m256)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_sub_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_sub_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(sub(cast<float>(l), cast<float>(r)));
                } else {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm256_sub_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm256_sub_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_sub_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm256_sub_epi64(l, r));
                    }
                    #endif
                }
            } else if constexpr (size * 2 == sizeof(__m256) && Merge) {
                return sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (size == sizeof(__m512)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_sub_ps(l, r));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_sub_pd(l, r));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(sub(cast<float>(l), cast<float>(r)));
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(_mm512_sub_epi8(l, r));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(_mm512_sub_epi16(l, r));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_sub_epi32(l, r));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_sub_epi64(l, r));
                    }
                }
            } else if constexpr (size * 2 == sizeof(__m512) && Merge) {
                return sub(from_vec<T>(fit_to_vec(lhs)), from_vec<T>(fit_to_vec(rhs))).lo;
            }
            #endif
            return join(
                sub<false>(lhs.lo, rhs.lo),
                sub<false>(lhs.hi, rhs.lo)
            );
        }
    }
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_SUB_HPP

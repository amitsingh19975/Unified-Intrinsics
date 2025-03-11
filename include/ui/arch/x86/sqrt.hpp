#ifndef AMT_UI_ARCH_X86_SQRT_HPP
#define AMT_UI_ARCH_X86_SQRT_HPP

#include "cast.hpp"
#include "../emul/sqrt.hpp"

namespace ui::x86 {

    template<bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sqrt(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::sqrt(v);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_sqrt_ps(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_sqrt_pd(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(sqrt(cast<float>(v)));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                return sqrt(from_vec<T>(fit_to_vec(v))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (bits == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_sqrt_ps(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_sqrt_pd(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(sqrt(cast<float>(v)));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_sqrt_ps(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_sqrt_pd(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(sqrt(cast<float>(v)));
                }
            }
            #endif

            return join(
                sqrt<false>(v.lo),
                sqrt<false>(v.hi)
            );
        }
    }
    
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_SQRT_HPP

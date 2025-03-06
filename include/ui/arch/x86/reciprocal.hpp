#ifndef AMT_UI_ARCH_X86_RECIPROCAL_HPP
#define AMT_UI_ARCH_X86_RECIPROCAL_HPP

#include "cast.hpp"
#include "mul.hpp"
#include "sub.hpp"
#include "../emul/reciprocal.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::x86 {

    template <bool Merge = true, std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::reciprocal_estimate(v);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_rcp_ps(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(reciprocal_estimate(cast<float>(v)));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                return reciprocal_estimate(from_vec<T>(fit_to_vec(v))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (bits == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_rcp_ps(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(reciprocal_estimate(cast<float>(v)));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_rcp_ps(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(reciprocal_estimate(cast<float>(v)));
                }
            }
            #endif

            return join(
                reciprocal_estimate<false>(v.lo),
                reciprocal_estimate<false>(v.hi)
            );
        }
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto reciprocal_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        if constexpr (std::floating_point<T>) {
            if constexpr (::ui::internal::is_fp16<T>) {
                auto v0 = cast<float>(v);
                auto e0 = cast<float>(e);
                auto f2 = Vec<N, float>::load(2);
                auto m = mul(v0, e0);
                auto temp = sub(f2, m);
                return cast<T>(temp);
            } else {
                auto f2 = Vec<N, T>::load(2);
                auto m = mul(v, e);
                return sub(f2, m); 
            }
        } else {
            return emul::reciprocal_refine(v, e);
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sqrt_inv_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::sqrt_inv_estimate(v);
        } else {
            if constexpr (bits == sizeof(__m128)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_rsqrt_ps(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(sqrt_inv_estimate(cast<float>(v)));
                }
            } else if constexpr (bits * 2 == sizeof(__m128) && Merge) {
                return sqrt_inv_estimate(from_vec<T>(fit_to_vec(v))).lo;
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (bits == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_rsqrt_ps(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(sqrt_inv_estimate(cast<float>(v)));
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_rsqrt_ps(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(sqrt_inv_estimate(cast<float>(v)));
                }
            }
            #endif

            return join(
                sqrt_inv_estimate<false>(v.lo),
                sqrt_inv_estimate<false>(v.hi)
            );
        }
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sqrt_inv_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        if constexpr (std::floating_point<T>) {
            if constexpr (::ui::internal::is_fp16<T>) {
                auto v0 = cast<float>(v);
                auto e0 = cast<float>(e);
                auto f3 = Vec<N, float>::load(3);
                auto f05 = Vec<N, float>::load(0.5);
                auto m = mul(v0, e0);
                f3 = sub(f3, m);
                return mul(f3, f05); 
            } else {
                auto f3 = Vec<N, T>::load(3);
                auto f05 = Vec<N, T>::load(0.5);
                auto m = mul(v, e);
                f3 = sub(f3, m);
                return mul(f3, f05); 
            }
        } else {
            return emul::sqrt_inv_refine(v, e);
        }
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T>)
    UI_ALWAYS_INLINE auto exponent_reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        return emul::exponent_reciprocal_estimate(v);
    }
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_RECIPROCAL_HPP

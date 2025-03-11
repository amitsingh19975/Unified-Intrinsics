#ifndef AMT_UI_ARCH_X86_MUL_HPP
#define AMT_UI_ARCH_X86_MUL_HPP

#include "cast.hpp"
#include "add.hpp"
#include "sub.hpp"
#include "../emul/mul.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::x86 {

// MARK: Multiplication
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::mul(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(__m128i)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm_mul_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm_mul_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(mul(cast<float>(lhs), cast<float>(rhs)));
                } else {
                    auto a = to_vec(lhs);
                    auto b = to_vec(rhs);
                    if constexpr (sizeof(T) == 1) {
                        auto even = _mm_mullo_epi16(a, b);
                        auto odd = _mm_mullo_epi16(
                            _mm_srli_epi16(a, 8),
                            _mm_srli_epi16(b, 8)
                        );
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                        return from_vec<T>(_mm_or_si128(
                            _mm_slli_epi16(odd, 8),
                            _mm_and_si128(even, _mm_set1_epi16(0xff))
                        ));
                        #else
                        return from_vec<T>(_mm_or_si128(
                            _mm_slli_epi16(odd, 8),
                            _mm_srli_epi16(
                                _mm_slli_epi16(even, 8),
                                8
                            )
                        ));
                        #endif
                    } else if constexpr (sizeof(T) == 2) {
                       return from_vec<T>(_mm_mullo_epi16(a, b)); 
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm_mullo_epi32(a, b));
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm_mullo_epi64(a, b));
                    #endif
                    }
                }
            } else if constexpr (bits * 2 == sizeof(__m128)) {
                auto tl = fit_to_vec(lhs);
                auto tr = fit_to_vec(rhs);
                if constexpr (sizeof(T) == 1) {
                    auto a = _mm_cvtepi8_epi16(tl);
                    auto b = _mm_cvtepi8_epi16(tr);
                    auto res = _mm_mullo_epi16(a, b);
                    if constexpr (std::is_signed_v<T>) {
                        res = _mm_shuffle_epi8(res, *reinterpret_cast<__m128i const*>(constants::mask8_16_even_odd));
                    } else {
                        auto mask = _mm_set1_epi16(static_cast<std::int16_t>(0xff));
                        res = _mm_and_si128(res, mask);
                        res = _mm_packus_epi16(res, res);
                    }
                    return from_vec<T>(res).lo;
                }
                return mul(
                    from_vec<T>(tl),
                    from_vec<T>(tr)
                ).lo;
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
            if constexpr (bits == sizeof(__m256)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm256_mul_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm256_mul_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(mul(cast<float>(lhs), cast<float>(rhs)));
                } else {
                    auto a = to_vec(lhs);
                    auto b = to_vec(rhs);
                    if constexpr (sizeof(T) == 1) {
                        auto even = _mm256_mullo_epi16(a, b);
                        auto odd = _mm256_mullo_epi16(
                            _mm256_srli_epi16(a, 8),
                            _mm256_srli_epi16(b, 8)
                        );
                        return from_vec<T>(_mm256_or_si256(
                            _mm256_slli_epi16(odd, 8),
                            _mm256_and_si256(even, _mm256_set1_epi16(0xff))
                        ));
                    } else if constexpr (sizeof(T) == 2) {
                       return from_vec<T>(_mm256_mullo_epi16(a, b)); 
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm256_mullo_epi32(a, b));
                    } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        return from_vec<T>(_mm256_mullo_epi64(a, b));
                    #else
                        auto b_swap = _mm256_shuffle_epi32(b, _MM_SHUFFLE(2,3, 0,1));
                        auto crossprod = _mm256_mullo_epi32(a, b_swap);
                        auto prodlh = _mm256_slli_epi64(crossprod, 32);
                        auto prodhl = _mm256_and_si256(crossprod, _mm256_set1_epi64x(0xFFFFFFFF00000000));
                        auto sumcross = _mm256_add_epi32(prodlh, prodhl);
                        auto prodll = _mm256_mul_epu32(a, b);
                        auto prod = _mm256_add_epi32(prodll, sumcross);
                        return from_vec<T>(prod);
                    #endif
                    }
                }
            }
            #endif

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(_mm512_mul_ps(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(_mm512_mul_pd(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(mul(cast<float>(lhs), cast<float>(rhs)));
                } else {
                    auto a = to_vec(lhs);
                    auto b = to_vec(rhs);
                    if constexpr (sizeof(T) == 1) {
                        auto even = _mm512_mullo_epi16(a, b);
                        auto odd = _mm512_mullo_epi16(
                            _mm512_srli_epi16(a, 8),
                            _mm512_srli_epi16(b, 8)
                        );
                        return from_vec<T>(_mm512_or_si1512(
                            _mm512_slli_epi16(odd, 8),
                            _mm512_and_si1512(even, _mm512_set1_epi16(0xff))
                        ));
                    } else if constexpr (sizeof(T) == 2) {
                       return from_vec<T>(_mm512_mullo_epi16(a, b)); 
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(_mm512_mullo_epi32(a, b));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(_mm512_mullo_epi64(a, b));
                    }
                }
            }
            #endif
            return join(
                mul(lhs.lo, rhs.lo),
                mul(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Multiply-Accumulate
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::add_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (::ui::internal::is_fp16<T>) {
            auto a = cast<float>(acc);
            auto l = cast<float>(lhs);
            auto r = cast<float>(rhs);
            return cast<T>(add(
                a,
                mul(l, r)
            ));
        } else {
            return add(
                acc,
                mul(lhs, rhs)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        [[maybe_unused]] op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (::ui::internal::is_fp16<T>) {
            auto a = cast<float>(acc);
            auto l = cast<float>(lhs);
            auto r = cast<float>(rhs);
            return cast<T>(sub(
                a,
                mul(l, r)
            ));
        } else {
            return sub(
                acc,
                mul(lhs, rhs)
            );
        }
    }

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <typename Op, std::size_t N, std::integral T>
        requires (sizeof(T) < 8 && (std::same_as<Op, op::add_t> || std::same_as<Op, op::sub_t>))
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        Op op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto l = cast<result_t>(lhs);
        auto r = cast<result_t>(rhs);
        return mul_acc(acc, l, r, op);
    }
// !MARK

// MARK: Fused-Multiply-Accumulate
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::add_t op
    ) noexcept -> Vec<N, T> {
    #ifdef UI_SUPPORT_FMA
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::fused_mul_acc(acc, lhs, rhs, op);
        } else {
            if constexpr (bits == sizeof(__m128i)) {
                auto a = to_vec(acc);
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm_fmadd_ps(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, double>) {
                    auto res = _mm_fmadd_pd(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                }
            } else if constexpr (bits * 2 == sizeof(__m128)) {
                auto tl = fit_to_vec(lhs);
                auto tr = fit_to_vec(rhs);
                return mul(
                    from_vec<T>(tl),
                    from_vec<T>(tr)
                ).lo;
            }

            if constexpr (bits == sizeof(__m256)) {
                auto a = to_vec(acc);
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm256_fmadd_ps(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, double>) {
                    auto res = _mm256_fmadd_pd(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                }
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                auto a = to_vec(acc);
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm512_fmadd_ps(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, double>) {
                    auto res = _mm512_fmadd_pd(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                }
            }
            #endif
            return join(
                fused_mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                fused_mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    #else
        return mul_acc(acc, lhs, rhs, op);
    #endif
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
    #ifdef UI_SUPPORT_FMA
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::fused_mul_acc(acc, lhs, rhs, op);
        } else {
            if constexpr (bits == sizeof(__m128i)) {
                auto a = to_vec(acc);
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm_fnmadd_ps(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, double>) {
                    auto res = _mm_fnmadd_pd(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                }
            } else if constexpr (bits * 2 == sizeof(__m128)) {
                auto tl = fit_to_vec(lhs);
                auto tr = fit_to_vec(rhs);
                return mul(
                    from_vec<T>(tl),
                    from_vec<T>(tr)
                ).lo;
            }

            if constexpr (bits == sizeof(__m256)) {
                auto a = to_vec(acc);
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm256_fnmadd_ps(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, double>) {
                    auto res = _mm256_fnmadd_pd(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                }
            }

            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (bits == sizeof(__m512)) {
                auto a = to_vec(acc);
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto res = _mm512_fnmadd_ps(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, double>) {
                    auto res = _mm512_fnmadd_pd(l, r, a);
                    return from_vec<T>(res);
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                }
            }
            #endif
            return join(
                fused_mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                fused_mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    #else
        return mul_acc(acc, lhs, rhs, op);
    #endif
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return fused_mul_acc(acc, a, temp, op);
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return fused_mul_acc(acc, a, temp, op);
    }
// !MARK

// MARK: Widening Multiplication
    template <std::size_t N, std::integral T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        auto l = cast<result_t>(lhs);
        auto r = cast<result_t>(rhs);
        return mul(l, r);
    }
// !MARK

// MARK: Vector multiply-accumulate by scalar
    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        Vec<M, T> const& v,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return mul_acc(a, b, temp, op);
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(v[Lane]);
        return mul_acc(a, b, temp, op);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(c);
        return mul_acc(a, b, temp, op);
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        auto temp = Vec<N, T>::load(c);
        return mul_acc(a, b, temp, op);
    }
// !MARK

// MARK: Multiplication with scalar
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, T> {
        return mul(v, Vec<N, T>::load(c));
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        return mul(a, Vec<N, T>::load(v[Lane]));
    }
// !MARK

// MARK: Multiplication with scalar and widen
    template <std::size_t N, typename T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul(cast<result_t>(v), static_cast<result_t>(c));
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul(cast<result_t>(a), static_cast<result_t>(v[Lane]));
    }
// !MARK

// MARK: Vector multiply-accumulate by scalar and widen
    template <std::size_t N, typename T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul_acc(
            a,
            cast<result_t>(v),
            Vec<N, result_t>::load(static_cast<result_t>(c)),
            op
        );
    }

    template <std::size_t N, typename T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        return mul_acc(
            a,
            cast<result_t>(v),
            Vec<N, result_t>::load(static_cast<result_t>(c)),
            op
        );
    }
// !MARK

// MARK: Fused multiply-accumulate by scalar
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        return fused_mul_acc(
            a,
            b,
            Vec<N, T>::load(c),
            op
        );
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        return fused_mul_acc(
            a,
            b,
            Vec<N, T>::load(c),
            op
        );
    }
// !MARK
} // namespace ui::x86

#endif // AMT_UI_ARCH_X86_MUL_HPP

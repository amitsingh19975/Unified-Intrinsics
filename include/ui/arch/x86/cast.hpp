#ifndef AMT_UI_ARC_X86_HPP
#define AMT_UI_ARC_X86_HPP

#include "../../base_vec.hpp"
#include "../../base.hpp"
#include "../basic.hpp"
#include "../../vec_headers.hpp"
#include "../../float.hpp"
#include "../../matrix.hpp"
#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <functional>
#include "basic.hpp"

namespace ui::x86 {

   template <std::size_t N, typename T>
   UI_ALWAYS_INLINE constexpr auto to_vec(Vec<N, T> const& v) noexcept {
      if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return std::bit_cast<__m128>(v);
                } else if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256>(v);
                    #endif
                } else if constexpr (N == 16) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512>(v);
                    #endif
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                    return std::bit_cast<__m128d>(v);
                    #endif
                } else if constexpr (N == 4) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256d>(v);
                    #endif
                } else if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512d>(v);
                    #endif
                }
            } else if constexpr (std::same_as<T, float16>) {
                if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                    return std::bit_cast<__m128i>(v);
                    #endif
                } else if constexpr (N == 16) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256i>(v);
                    #endif
                } else if constexpr (N == 32) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512i>(v);
                    #endif
                }
            } else if constexpr (std::same_as<T, bfloat16>) {
                if constexpr (N == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                    return std::bit_cast<__m128i>(v);
                    #endif
                } else if constexpr (N == 16) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    return std::bit_cast<__m256i>(v);
                    #endif
                } else if constexpr (N == 32) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    return std::bit_cast<__m512i>(v);
                    #endif
                }
            } else {
                static_assert(sizeof(T) == sizeof(float) || sizeof(T) == sizeof(double) || sizeof(T) == sizeof(float16) || sizeof(T) == sizeof(bfloat16), "Unknow floating-point type, expecting 'float', 'ui::float16', 'ui::bfloat16' or 'double'");
            }
        } else {
            static constexpr auto bits = N * sizeof(T) * 8;
            if constexpr (bits == 128) {
                return std::bit_cast<__m128i>(v);
            } else if constexpr (bits == 256) {
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                return std::bit_cast<__m256i>(v);
                #endif
            } else if constexpr (bits == 512) {
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                return std::bit_cast<__m512i>(v);
                #endif
            } else {
                static_assert(bits >= 128, "N * sizeof(T) * 8 must be at least 128 bits");
                static_assert(bits <= 512, "N * sizeof(T) * 8 must be at most 512 bits");
            }
        } 
   }

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
    template <typename T, std::size_t N = sizeof(__m128i) / sizeof(T)>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m128i v) noexcept -> Vec<N, T> {
        static_assert(std::integral<T> || std::same_as<T, float16> || std::same_as<T, bfloat16>, "cannot convvert to unreleated types");
        return std::bit_cast<Vec<N, T>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
    template <typename T, std::size_t N = sizeof(__m256i) / sizeof(T)>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m256i v) noexcept -> Vec<N, T> {
        static_assert(std::integral<T> || std::same_as<T, float16> || std::same_as<T, bfloat16>, "cannot convvert to unreleated types");
        return std::bit_cast<Vec<N, T>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
    template <typename T, std::size_t N = sizeof(__m512i) / sizeof(T)>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m512i v) noexcept -> Vec<N, T> {
        static_assert(std::integral<T> || std::same_as<T, float16> || std::same_as<T, bfloat16>, "cannot convvert to unreleated types");
        return std::bit_cast<Vec<N, T>>(v); 
    }
    #endif

    template <std::same_as<float> T>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m128 v) noexcept -> Vec<4, T> {
        return std::bit_cast<Vec<4, T>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m128 v) noexcept -> Vec<4, float> {
        return std::bit_cast<Vec<4, float>>(v); 
    }

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
    template <std::same_as<float> T>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m256 v) noexcept -> Vec<8, float> {
        return std::bit_cast<Vec<8, float>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m256 v) noexcept -> Vec<8, float> {
        return std::bit_cast<Vec<8, float>>(v); 
    }

    template <std::same_as<double> T>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m256d v) noexcept -> Vec<4, double> {
        return std::bit_cast<Vec<4, double>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m256d v) noexcept -> Vec<4, double> {
        return std::bit_cast<Vec<4, double>>(v); 
    }
    #endif

    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
    template <std::same_as<float> T>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m256 v) noexcept -> Vec<16, float> {
        return std::bit_cast<Vec<16, float>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m256 v) noexcept -> Vec<16, float> {
        return std::bit_cast<Vec<16, float>>(v); 
    }

    template <std::same_as<double> T>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m512d v) noexcept -> Vec<8, double> {
        return std::bit_cast<Vec<8, double>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m512d v) noexcept -> Vec<8, double> {
        return std::bit_cast<Vec<8, double>>(v); 
    }
    #endif

    namespace internal {
        template <std::size_t N, typename Fn>
        struct Case {
            static constexpr auto pattern = N;
            template <typename... Args>
            constexpr decltype(auto) operator()(Args&&... args) const noexcept 
                requires std::invocable<Fn, Args...>
            {
                return std::invoke(fn, std::forward<Args>(args)...);
            }

            template <typename Tn>
            constexpr Case<N, Tn> operator=(Tn&& f) const noexcept {
                return { .fn = std::forward<Tn>(f) };
            }

            Fn fn;
        };

        template <std::size_t N>
        static constexpr auto case_maker = Case<N, void*>(nullptr);

        template <typename... Cs>
        struct Matcher {
            using base_type = std::tuple<Cs...>;
            base_type cases;

            template <std::size_t P>
            static constexpr auto can_match_case = ((P == Cs::pattern) || ...);

            constexpr Matcher() noexcept = default;
            constexpr Matcher(Cs&&... cs) noexcept
                : cases(std::forward<Cs>(cs)...)
            {}
            constexpr Matcher(Matcher const& fn) noexcept = default;
            constexpr Matcher(Matcher && fn) noexcept = default;
            constexpr Matcher& operator=(Matcher const& fn) noexcept = default;
            constexpr Matcher& operator=(Matcher && fn) noexcept = default;
            constexpr ~Matcher() noexcept = default;

            template <std::size_t P, typename... Args>
                requires can_match_case<P>
            constexpr decltype(auto) match(Args&&... args) const noexcept {
                return invoke_helper<0, P>(std::forward<Args>(args)...); 
            }


        private:
            template <std::size_t I = 0, std::size_t P, typename... Args>
            constexpr decltype(auto) invoke_helper(Args&&... args) const noexcept {
                if constexpr (sizeof... (Cs) <= I) return;
                else if constexpr (std::tuple_element_t<I, base_type>::pattern == P) {
                    return std::invoke(std::get<I>(cases), std::forward<Args>(args)...);
                } else {
                    return invoke_helper<I + 1, P>(std::forward<Args>(args)...);
                }
            }
        };

        template<typename... Ts>
        Matcher(Ts...) -> Matcher<Ts...>;

        template<typename... Ts>
        struct Overloaded: Ts... { using Ts::operator()...; };

        template<typename... Ts>
        Overloaded(Ts...) -> Overloaded<Ts...>;

        template <typename T>
        concept not_void = !std::is_void_v<T>;

        template <std::size_t P, typename M, typename... Args>
        concept is_case_invocable = (M::template can_match_case<P>) && requires (M const& m) {
            { m.template match<P>(std::declval<Args>()...) } -> not_void;
        };

        template <typename To, typename M, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto cast_iter_chunk(Vec<N, T> const& v, M const& m) noexcept -> Vec<N, To> {
            if constexpr (N == 1) {
                std::println("Base Case of N == 1");
               if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                  return Vec<N, To>{ .val = static_cast<To>(float(v.val)) };
               } else {
                  return Vec<N, To>{ .val = static_cast<To>(v.val) };
               }
            } else {
                if constexpr (is_case_invocable<N, M, decltype(v)>) {
                    return m.template match<N>(v);
                } else {
                    return join(cast_iter_chunk<To>(v.lo, m), cast_iter_chunk<To>(v.hi, m));
                }
            }
        }

        template <typename To, typename Fn, std::size_t N, typename T>
            requires (
                N * sizeof(T) * 8 <= 512 &&
                N * sizeof(T) * 8 >= 128 &&
                !std::is_void_v<decltype(to_vec(Vec<N, T>{}))> &&
                std::invocable<Fn, decltype(to_vec(Vec<N, T>{}))>
            )
        UI_ALWAYS_INLINE auto cast_helper(Vec<N, T> const& vt, Fn&& fn) noexcept -> Vec<N, To> {
            auto v = to_vec(vt);
            using ret_t = std::decay_t<decltype(fn(v))>;

            constexpr auto shift = [](auto v_, int s) {
            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                return _mm_srli_si128(v_, s);
            #elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                if constexpr (sizeof(__m256) == sizeof(v)) {
                    return _mm256_srli_si256(v_, s);
                } else {
                    return _mm_srli_si128(v_, s);
                }
            #elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(__m512) == sizeof(v)) {
                    return _mm512_srli_si512(v_, s);
                } else if constexpr (sizeof(__m256) == sizeof(v)) {
                    return _mm256_srli_si256(v_, s);
                } else {
                    return _mm_srli_si128(v_, s);
                }
            #endif
            };

            if constexpr (sizeof(T) <= sizeof(To)) {
                static constexpr auto unroll = (sizeof(To) * N) / sizeof(ret_t);
                static constexpr auto ratio = sizeof(ret_t) / sizeof(To);

                if constexpr (unroll <= 1) {
                    return from_vec<To>(fn(v));
                } else if constexpr (unroll == 2) {
                    auto t0 = from_vec<To>(fn(v));
                    auto t1 = from_vec<To>(fn(shift(v, ratio)));
                    return join(t0, t1);
                } else if constexpr (unroll == 4) {
                    auto t0 = v;
                    auto t1 = shift(t0, ratio);
                    auto t2 = shift(t1, ratio);
                    auto t3 = shift(t2, ratio);
                    auto v0 = from_vec<To>(fn(t0));
                    auto v1 = from_vec<To>(fn(t1));
                    auto v2 = from_vec<To>(fn(t2));
                    auto v3 = from_vec<To>(fn(t3));

                    return join(
                        join(v0, v1),
                        join(v2, v3)
                    );
                } else {
                    constexpr auto helper = [&v, &fn, shift]<std::size_t... Is>(std::index_sequence<Is...>) {
                        return join(from_vec<To>(fn(v)), (from_vec<To>(fn(shift(v, ratio * Is))))...);
                    };
                    return helper(std::make_index_sequence<unroll>{});
                }
            } else {
                return from_vec<To>(fn(v));
            }
        }

        template <typename To, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto cast_half_helper(Vec<N, T> const& v, auto&& fn) noexcept -> Vec<N, To> {
            auto vt = Vec<N << 1, T>(v, v);
            return cast_helper<To>(vt, fn).lo;
        }


        template <typename T>
        struct is_native_vec: std::bool_constant<
            sizeof(T) * 8 == 128 || 
            sizeof(T) * 8 == 256 || 
            sizeof(T) * 8 == 512
        > {};

        template <typename T>
        concept native_vec = is_native_vec<T>::value;

        template <std::size_t N, typename T, typename To = std::make_unsigned_t<T>>
            requires (std::is_signed_v<T>)
        UI_ALWAYS_INLINE auto signed_to_unsigned_saturating_cast_fn(Vec<N, T> const& v) noexcept -> Vec<N, To> {
            auto m = to_vec(v);
            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX2
            auto zeros = _mm_setzero_si128();
            return from_vec<To>(_mm_max_epi16(zeros, m));
            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
            auto zeros = _mm256_setzero_si256();
            return from_vec<To>(_mm256_max_epi16(zeros, m));
            #else
            auto zeros = _mm512_setzero_si512();
            return from_vec<To>(_mm512_max_epi16(zeros, m));
            #endif
        }

        template <std::size_t N, typename T, typename To = std::make_unsigned_t<T>>
            requires (std::is_signed_v<T>)
        UI_ALWAYS_INLINE auto saturating_cast(Vec<N, T> const& v) noexcept -> Vec<N, To> {
            constexpr auto fn = [](auto const& v_) {
                return signed_to_unsigned_saturating_cast_fn(v_);
            };
            return cast_iter_chunk<To>(
                 v,
                 Matcher {
                    case_maker<4> = [fn](auto const& v_) {
                        return fn(join(v_, v_));
                    },
                    case_maker<8> = fn
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    , case_maker<16> = fn
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    , case_maker<32> = fn
                    #endif
                }
             );
        }

        template <std::size_t N, typename T, typename To = std::make_signed_t<T>>
            requires (!std::is_signed_v<T>)
        UI_ALWAYS_INLINE auto unsigned_to_signed_saturating_cast_fn(Vec<N, T> const& v) noexcept -> Vec<N, To> {
            auto m = to_vec(v);
            static constexpr auto max = std::numeric_limits<T>::max();
            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX2
            auto mx = _mm_set1_epi16(max);
            return from_vec<To>(_mm_min_epi16(mx, m));
            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
            auto mx = _mm_set1_epi16(max);
            return from_vec<To>(_mm256_min_epi16(mx, m));
            #else
            auto mx = _mm_set1_epi16(max);
            return from_vec<To>(_mm512_min_epi16(mx, m));
            #endif
        }

        template <std::size_t N, typename T, typename To = std::make_signed_t<T>>
            requires (!std::is_signed_v<T>)
        UI_ALWAYS_INLINE auto saturating_cast(Vec<N, T> const& v) noexcept -> Vec<N, To> {
            constexpr auto fn = [](auto const& v_) {
                return unsigned_to_signed_saturating_cast_fn(v_);
            };
            return cast_iter_chunk<To>(
                 v,
                 Matcher {
                    case_maker<4> = [fn](auto const& v_) {
                        return fn(join(v_, v_));
                    },
                    case_maker<8> = fn
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    , case_maker<16> = fn
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    , case_maker<32> = fn
                    #endif
                }
             );
        }

        template <std::size_t N, typename T>
            requires (!std::is_signed_v<T>)
        UI_ALWAYS_INLINE auto convert_unsigned_to_float(
            Vec<N, T> const& v
        ) noexcept -> Vec<N, float> {
            auto m = to_vec(v);
            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX2
                #define SHIFT(V) _mm_srli_epi32(V, sizeof(T) * 8 - 1)
                #define BROADCAST(V) _mm_set1_epi32(V)
                #define BROADCAST_F(V) _mm_set1_ps(V)
                #define AND(L, R) _mm_and_si128(L, R)
                #define F(I) _mm_cvtepi32_ps(I)
                #define MUL(L, R) _mm_mul_ps(L, R)
                #define ADD(L, R) _mm_add_ps(L, R)
            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                #define SHIFT(V) _mm256_srli_epi32(V, sizeof(T) * 8 - 1)
                #define BROADCAST(V) _mm256_set1_epi32(V)
                #define BROADCAST_F(V) _mm256_set1_ps(V)
                #define AND(L, R) _mm256_and_si128(L, R)
                #define F(I) _mm256_cvtepi32_ps(I)
                #define MUL(L, R) _mm256_mul_ps(L, R)
                #define ADD(L, R) _mm256_add_ps(L, R)
            #else
                #define SHIFT(V) _mm512_srli_epi32(V, sizeof(T) * 8 - 1)
                #define BROADCAST(V) _mm512_set1_epi32(V)
                #define BROADCAST_F(V) _mm512_set1_ps(V)
                #define AND(L, R) _mm512_and_si128(L, R)
                #define F(I) _mm512_cvtepi32_ps(I)
                #define MUL(L, R) _mm512_mul_ps(L, R)
                #define ADD(L, R) _mm512_add_ps(L, R)
            #endif

            auto high = SHIFT(m);
            static constexpr auto low_mask_val = std::numeric_limits<T>::max() >> 1;
            auto low_mask = BROADCAST(low_mask_val);
            auto low = AND(m, low_mask);
            auto base = F(low);
            auto high_f = F(high);
            auto offset = MUL(high_f, BROADCAST_F(low_mask_val));
            return from_vec<float>(ADD(base, offset));
            
            #undef SHIFT
            #undef BROADCAST
            #undef BROADCAST_F
            #undef AND
            #undef F
            #undef MUL
            #undef ADD
        }

        template <std::size_t N, typename T>
            requires (!std::is_signed_v<T>)
        UI_ALWAYS_INLINE auto convert_unsigned_to_double(
            Vec<N, T> const& v
        ) noexcept -> Vec<N, double> {
            auto m = to_vec(v);
            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX2
                #define SHIFT(V) _mm_srli_epi32(V, sizeof(T) * 8 - 1)
                #define BROADCAST(V) _mm_set1_epi32(V)
                #define BROADCAST_F(V) _mm_set1_pd(V)
                #define AND(L, R) _mm_and_si128(L, R)
                #define F(I) _mm_cvtepi32_pd(I)
                #define MUL(L, R) _mm_mul_pd(L, R)
                #define ADD(L, R) _mm_add_pd(L, R)
            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                #define SHIFT(V) _mm256_srli_epi32(V, sizeof(T) * 8 - 1)
                #define BROADCAST(V) _mm256_set1_epi32(V)
                #define BROADCAST_F(V) _mm256_set1_pd(V)
                #define AND(L, R) _mm256_and_si128(L, R)
                #define F(I) _mm256_cvtepi32_pd(I)
                #define MUL(L, R) _mm256_mul_pd(L, R)
                #define ADD(L, R) _mm256_add_pd(L, R)
            #else
                #define SHIFT(V) _mm512_srli_epi32(V, sizeof(T) * 8 - 1)
                #define BROADCAST(V) _mm512_set1_epi32(V)
                #define BROADCAST_F(V) _mm512_set1_pd(V)
                #define AND(L, R) _mm512_and_si128(L, R)
                #define F(I) _mm512_cvtepi32_pd(I)
                #define MUL(L, R) _mm512_mul_pd(L, R)
                #define ADD(L, R) _mm512_add_pd(L, R)
            #endif

            auto high = SHIFT(m);
            static constexpr auto low_mask_val = std::numeric_limits<T>::max() >> 1;
            auto low_mask = BROADCAST(low_mask_val);
            auto low = AND(m, low_mask);
            auto base = F(low);
            auto high_f = F(high);
            auto offset = MUL(high_f, BROADCAST_F(low_mask_val));
            return from_vec<double>(ADD(base, offset));
            
            #undef SHIFT
            #undef BROADCAST
            #undef BROADCAST_F
            #undef AND
            #undef F
            #undef MUL
            #undef ADD
        }


        template <typename To, bool Saturating = false>
        struct CastImpl {
            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int8_t> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t>{}(v);
                    auto os = Overloaded {
                        [](__m128i m) { 
                            return _mm_cvtepi32_ps(m);
                        }
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                        , [](__m256i m) { 
                            return _mm256_cvtepi32_ps(m);
                        }
                        #endif
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        , [](__m512i m) { 
                            return _mm512_cvtepi32_ps(m);
                        }
                        #endif
                    };
                   return cast_iter_chunk<To>(
                        temp,
                        Matcher {
                            case_maker<2> = [os](auto const& v_) {
                                return cast_half_helper<To>(v_, os);
                            },
                            case_maker<4> = [os](auto const& v_) {
                                return cast_helper<To>(v_, os);
                            }
                        }
                   );
                } else if constexpr (std::same_as<To, double>) {
                   auto temp = CastImpl<std::int32_t>{}(v);
                   return cast_iter_chunk<To>(
                        temp,
                        Matcher {
                            case_maker<2> = [](auto const& v_) {
                                return cast_half_helper<To>(v_, [](native_vec auto m) {
                                    #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                    return _mm_cvtepi32_pd(m);
                                    #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                    return _mm256_cvtepi32_pd(m);
                                    #else
                                    return _mm512_cvtepi32_pd(m);
                                    #endif
                                });
                            },
                            case_maker<4> = [](auto const& v_) {
                                return cast_helper<To>(v_, [](native_vec auto m) {
                                    #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                    return _mm_cvtepi32_pd(m);
                                    #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                    return _mm256_cvtepi32_pd(m);
                                    #else
                                    return _mm512_cvtepi32_pd(m);
                                    #endif
                                });
                            }
                        }
                   );   
                } else if constexpr (std::is_signed_v<To>) {
                   if constexpr (sizeof(To) == 1) {
                      return v;
                   } else if constexpr (sizeof(To) == 2) {
                         return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    return cast_half_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepi8_epi16(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepi8_epi16(m);
                                        #endif
                                    });
                                },
                                case_maker<16> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepi8_epi32(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepi8_epi16(m);
                                        #endif
                                    });
                                }
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<32> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m256i m) {
                                        return _mm512_cvtepi8_epi16(m);
                                    });
                                }
                                #endif
                             }
                         );
                   } else if constexpr (sizeof(To) == 4) {
                         return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    return cast_half_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepi8_epi32(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepi8_epi32(m);
                                        #else
                                        return _mm512_cvtepi8_epi32(m);
                                        #endif
                                    });
                                },
                                case_maker<16> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepi8_epi32(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepi8_epi32(m);
                                        #else
                                        return _mm512_cvtepi8_epi32(m);
                                        #endif
                                    });
                                }
                             }
                         );
                   } else if constexpr (sizeof(To) == 8) {
                         return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    return cast_half_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepi8_epi64(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepi8_epi64(m);
                                        #else
                                        return _mm512_cvtepi8_epi64(m);
                                        #endif
                                    });
                                },
                                case_maker<16> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepi8_epi64(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepi8_epi64(m);
                                        #else
                                        return _mm512_cvtepi8_epi64(m);
                                        #endif
                                    });
                                }
                             }
                         );
                   }

                } else {
                    if constexpr (sizeof(To) == 1 && Saturating) {
                        return saturating_cast(v);
                    }
                    return std::bit_cast<Vec<N, To>>(CastImpl<std::make_signed_t<To>>{}(v));
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint8_t> const& v      
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t>{}(v);
                    return CastImpl<float>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<std::int32_t>{}(v);
                    return CastImpl<double>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1 && Saturating) {
                        return saturating_cast(v);
                    }
                    auto temp = CastImpl<std::make_unsigned_t<To>, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else {
                    if constexpr (sizeof(To) == 1) return v;
                    else if constexpr (sizeof(To) == 2) {
                         return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    return cast_half_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepu8_epi16(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepu8_epi16(m);
                                        #endif
                                    });
                                },
                                case_maker<16> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepu8_epi32(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepu8_epi16(m);
                                        #endif
                                    });
                                }
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<32> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m256i m) {
                                        return _mm512_cvtepu8_epi16(m);
                                    });
                                }
                                #endif
                             }
                         );
                    } else if constexpr (sizeof(To) == 4) {
                        return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    return cast_half_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepu8_epi32(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepu8_epi32(m);
                                        #else
                                        return _mm512_cvtepu8_epi32(m);
                                        #endif
                                    });
                                },
                                case_maker<16> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepu8_epi32(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepu8_epi32(m);
                                        #else
                                        return _mm512_cvtepu8_epi32(m);
                                        #endif
                                    });
                                }
                             }
                         );
                    } else if constexpr (sizeof(To) == 8) {
                        return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<8> = [](auto const& v_) {
                                    return cast_half_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepu8_epi64(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepu8_epi64(m);
                                        #else
                                        return _mm512_cvtepu8_epi64(m);
                                        #endif
                                    });
                                },
                                case_maker<16> = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](__m128i m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_cvtepu8_epi64(m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm256_cvtepu8_epi64(m);
                                        #else
                                        return _mm512_cvtepu8_epi64(m);
                                        #endif
                                    });
                                }
                             }
                         );
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int16_t> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t>{}(v);
                    return CastImpl<float>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<std::int32_t>{}(v);
                    return CastImpl<double>{}(temp);
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (Saturating) {
                                return from_vec<To>(_mm_packs_epi16(m, m)).lo;
                            } else {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                auto temp = _mm_shuffle_epi8(m, *reinterpret_cast<__m128i const*>(constants::mask8_16_even_odd));
                                return from_vec<To>(temp).lo;
                            #else
                                return from_vec<To>(_mm_cvtepi16_epi8(m)).lo;
                            #endif
                            } 
                        };
                        return cast_iter_chunk<To>(
                            v,
                            Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_,v_)).lo;
                                },
                                case_maker<8> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                                , case_maker<16> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        auto vec = from_vec<To>(_mm256_packs_epi16(m, m));
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    } else {
                                    #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        auto temp = _mm256_shuffle_epi8(m, *reinterpret_cast<__m256i const*>(constants::mask8_16_even_odd));
                                        auto vec = from_vec<To>(temp);
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    #else
                                        return from_vec<To>(_mm256_cvtepi16_epi8(m));
                                    #endif
                                    } 
                                }
                                #endif
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<32> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        return from_vec<To>(_mm512_packs_epi16(m, m)).lo;
                                    } else {
                                        return from_vec<To>(_mm512_cvtepi16_epi8(m));
                                    } 
                                }
                                #endif
                            }
                        );
                    } else if constexpr (sizeof(To) == 2) {
                        return v;
                    } else if constexpr (sizeof(To) == 4) {
                        constexpr auto fn = [](auto const& v_) {
                            return cast_helper<To>(v_, [](__m128i m) {
                                #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                return _mm_cvtepu16_epi32(m);
                                #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                return _mm256_cvtepu16_epi32(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<8> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<16> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    return _mm512_cvtepu16_epi32(m); 
                                }
                                #endif
                            }
                         );
                    } else if constexpr (sizeof(To) == 8) {
                        constexpr auto fn = [](auto const& v_) {
                            return cast_helper<To>(v_, [](__m128i m) {
                                #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                return _mm_cvtepu16_epi64(m);
                                #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                return _mm256_cvtepu16_epi64(m);
                                #else
                                return _mm512_cvtepu16_epi64(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<8> = fn
                            }
                         );
                    }
                } else {
                    if constexpr (Saturating) {
                        if constexpr (sizeof(To) == 1) {
                            constexpr auto fn = [](auto const& v_) {
                                auto m = to_vec(v_);
                                return from_vec<To>(_mm_packus_epi16(m, m)).lo; 
                            };
                            return cast_iter_chunk<To>(
                                v,
                                Matcher {
                                    case_maker<4> = [fn](auto const& v_) {
                                        return fn(join(v_,v_)).lo;
                                    },
                                    case_maker<8> = fn
                                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                                    , case_maker<16> = [](auto const& v_) {
                                        auto m = to_vec(v_);
                                        auto vec = from_vec<To>(_mm256_packus_epi16(m, m));
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    }
                                    #endif
                                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                    , case_maker<32> = [](auto const& v_) {
                                        auto m = to_vec(v_);
                                        return from_vec<To>(_mm512_packus_epi16(m, m)).lo;
                                    }
                                    #endif
                                }
                            );

                        } else if constexpr (sizeof(To) == 2) {
                            return saturating_cast(v);
                        } else {
                            auto temp = CastImpl<std::make_signed_t<To>, Saturating>{}(v);
                            return std::bit_cast<Vec<N, To>>(temp);
                        }
                    } else {
                        auto temp = CastImpl<std::make_signed_t<To>, Saturating>{}(v);
                        return std::bit_cast<Vec<N, To>>(temp);
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint16_t> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t>{}(v);
                    return CastImpl<float>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<std::int32_t>{}(v);
                    return CastImpl<double>{}(temp);
                } else if constexpr (!std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 2) return v;
                    auto temp = CastImpl<std::make_signed_t<To>, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else {
                    if constexpr (Saturating) {
                        if constexpr (sizeof(To) == 1) {
                            constexpr auto fn = [](auto const& v_) {
                                auto m = to_vec(v_);
                                auto val_mask = _mm_set1_epi16(0x7fff);
                                auto trunc = _mm_and_si128(m, val_mask); // m & 0x7fff
                                auto masked_trunc = _mm_cmpgt_epi16(m, trunc); // m > trunc
                                masked_trunc = _mm_and_si128(trunc, masked_trunc); // trunc & masked_trunc(=0xffff if true else 0x0000)
                                trunc = _mm_or_si128(masked_trunc, trunc);
                                return from_vec<To>(_mm_packus_epi16(trunc, trunc)).lo;
                            };
                            return cast_iter_chunk<To>(
                                v,
                                Matcher {
                                    case_maker<4> = [fn](auto const& v_) {
                                        return fn(join(v_,v_)).lo;
                                    },
                                    case_maker<8> = fn
                                }
                            );
                        } else if constexpr (sizeof(To) == 2 && Saturating) {
                            return saturating_cast(v);
                        }
                    }
                    auto temp = CastImpl<To, Saturating>{}(std::bit_cast<Vec<N, std::int16_t>>(v));
                    return std::bit_cast<Vec<N, To>>(temp);
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int32_t> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                            return _mm_cvtepi32_ps(m);
                            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                            return _mm256_cvtepi32_ps(m);
                            #else
                            return _mm512_cvtepi32_ps(m);
                            #endif
                        });
                    };
                    return cast_iter_chunk<To>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            , case_maker<8> = fn
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<16> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::same_as<To, double>) {
                    constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                            return _mm_cvtepi32_pd(m);
                            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                            return _mm256_cvtepi32_pd(m);
                            #else
                            return _mm512_cvtepi32_pd(m);
                            #endif
                        });
                    };
                    return cast_iter_chunk<To>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            , case_maker<8> = fn
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<16> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<std::int8_t, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (Saturating) {
                                return from_vec<To>(_mm_packs_epi32(m, m)).lo;
                            } else {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                auto temp = _mm_shuffle_epi8(m, *reinterpret_cast<__m128i const*>(constants::mask8_32_even_odd));
                                return from_vec<To>(temp).lo;
                            #else
                                return from_vec<To>(_mm_cvtepi32_epi16(m)).lo;
                            #endif
                            } 
                        };
                        return cast_iter_chunk<To>(
                            v,
                            Matcher {
                                case_maker<2> = [fn](auto const& v_) {
                                    return fn(join(v_,v_)).lo;
                                },
                                case_maker<4> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                                , case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        auto vec = from_vec<To>(_mm256_packs_epi32(m, m));
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    } else {
                                    #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        auto temp = _mm256_shuffle_epi8(m, *reinterpret_cast<__m256i const*>(constants::mask8_32_even_odd));
                                        auto vec = from_vec<To>(temp);
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    #else
                                        return from_vec<To>(_mm256_cvtepi32_epi16(m));
                                    #endif
                                    } 
                                }
                                #endif
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<16> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        return from_vec<To>(_mm512_packs_epi32(m, m)).lo;
                                    } else {
                                        return from_vec<To>(_mm512_cvtepi32_epi16(m));
                                    } 
                                }
                                #endif
                            }
                        );
                    } else if constexpr (sizeof(To) == 4) {
                        return v;
                    } else if constexpr (sizeof(To) == 8) {
                        constexpr auto fn = [](auto const& v_) {
                            return cast_helper<To>(v_, [](__m128i m) {
                                #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                return _mm_cvtepi32_epi64(m);
                                #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                return _mm256_cvtepi32_epi64(m);
                                #else
                                return _mm512_cvtepi32_epi64(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<8> = fn
                            }
                         );
                    }
                } else {
                    if constexpr (Saturating) {
                        if constexpr (sizeof(To) == 1) {
                            auto temp = CastImpl<std::uint16_t>{}(v); 
                            return CastImpl<std::uint8_t>{}(temp);
                        } else if constexpr (sizeof(To) == 2) {
                            if constexpr (Saturating) {
                                constexpr auto fn = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](auto m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_packus_epi32(m, m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm_packus_epi32(m, m);
                                        #else
                                        return _mm_packus_epi32(m, m);
                                        #endif
                                    });
                                };
                                return cast_iter_chunk<To>(
                                     v,
                                     Matcher {
                                        case_maker<4> = [fn](auto const& v_) {
                                            return fn(join(v_, v_));
                                        },
                                        case_maker<8> = fn
                                    }
                                 );
                            }
                        } else if (sizeof(To) == 4 && Saturating) {
                            return saturating_cast(v);
                        }
                    }
                    auto temp = CastImpl<std::make_signed_t<To>, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint32_t> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    constexpr auto fn = [](auto const& v_) {
                        return convert_unsigned_to_float(v_);
                    };
                    return cast_iter_chunk<To>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            , case_maker<8> = fn
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<16> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::same_as<To, double>) {
                    constexpr auto fn = [](auto const& v_) {
                        return convert_unsigned_to_double(v_);
                    };
                    return cast_iter_chunk<To>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            , case_maker<8> = fn
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<16> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (Saturating) {
                        if constexpr (sizeof(To) == 4) {
                            return saturating_cast(v);
                        }
                    }
                    auto temp = CastImpl<std::make_signed<To>, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                        return CastImpl<To>{}(v);
                    } else if constexpr (sizeof(To) == 2) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (Saturating) {
                                return from_vec<To>(_mm_packus_epi32(m, m)).lo;
                            } else {
                                auto temp = _mm_shuffle_epi8(m, *reinterpret_cast<__m128i const*>(constants::mask8_32_even_odd));
                                return from_vec<To>(temp).lo;
                            } 
                        };
                        return cast_iter_chunk<To>(
                            v,
                            Matcher {
                                case_maker<2> = [fn](auto const& v_) {
                                    return fn(join(v_,v_)).lo;
                                },
                                case_maker<4> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                                , case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        auto vec = from_vec<To>(_mm256_packus_epi32(m, m));
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    } else {
                                        auto temp = _mm256_shuffle_epi8(m, *reinterpret_cast<__m256i const*>(constants::mask8_32_even_odd));
                                        auto vec = from_vec<To>(temp);
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    } 
                                }
                                #endif
                            }
                        );

                    } else if constexpr (sizeof(To) == 4) {
                        return v;
                    } else if constexpr (sizeof(To) == 8) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                            return _mm_cvtepu32_epi64(m);
                            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                            return _mm256_cvtepu32_epi64(m);
                            #else
                            return _mm512_cvtepu32_epi64(m);
                            #endif
                        };
                        return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<2> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<4> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                                , case_maker<8> = fn
                                #endif
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<16> = fn
                                #endif
                            }
                         );
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int64_t> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                            return _mm_cvtepi32_ps(m);
                            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                            return _mm256_cvtepi32_ps(m);
                            #else
                            return _mm512_cvtepi32_ps(m);
                            #endif
                        });
                    };
                    return cast_iter_chunk<To>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            , case_maker<8> = fn
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<16> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::same_as<To, double>) {
                    constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                            return _mm_cvtepi32_pd(m);
                            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                            return _mm256_cvtepi32_pd(m);
                            #else
                            return _mm512_cvtepi32_pd(m);
                            #endif
                        });
                    };
                    return cast_iter_chunk<To>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            , case_maker<8> = fn
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<16> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 1) {
                        auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                        return CastImpl<std::int8_t, Saturating>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (Saturating) {
                                return from_vec<To>(_mm_packs_epi32(m, m)).lo;
                            } else {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                auto temp = _mm_shuffle_epi8(m, *reinterpret_cast<__m128i const*>(constants::mask8_32_even_odd));
                                return from_vec<To>(temp).lo;
                            #else
                                return from_vec<To>(_mm_cvtepi32_epi16(m)).lo;
                            #endif
                            } 
                        };
                        return cast_iter_chunk<To>(
                            v,
                            Matcher {
                                case_maker<2> = [fn](auto const& v_) {
                                    return fn(join(v_,v_)).lo;
                                },
                                case_maker<4> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                                , case_maker<8> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        auto vec = from_vec<To>(_mm256_packs_epi32(m, m));
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    } else {
                                    #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        auto temp = _mm256_shuffle_epi8(m, *reinterpret_cast<__m256i const*>(constants::mask8_32_even_odd));
                                        auto vec = from_vec<To>(temp);
                                        return join(
                                            vec.lo.lo,
                                            vec.hi.lo
                                        );
                                    #else
                                        return from_vec<To>(_mm256_cvtepi32_epi16(m));
                                    #endif
                                    } 
                                }
                                #endif
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<16> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        return from_vec<To>(_mm512_packs_epi32(m, m)).lo;
                                    } else {
                                        return from_vec<To>(_mm512_cvtepi32_epi16(m));
                                    } 
                                }
                                #endif
                            }
                        );
                    } else if constexpr (sizeof(To) == 4) {
                        return v;
                    } else if constexpr (sizeof(To) == 8) {
                        constexpr auto fn = [](auto const& v_) {
                            return cast_helper<To>(v_, [](__m128i m) {
                                #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                return _mm_cvtepi32_epi64(m);
                                #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                return _mm256_cvtepi32_epi64(m);
                                #else
                                return _mm512_cvtepi32_epi64(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To>(
                             v,
                             Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<8> = fn
                            }
                         );
                    }
                } else {
                    if constexpr (Saturating) {
                        if constexpr (sizeof(To) == 1) {
                            auto temp = CastImpl<std::uint16_t>{}(v); 
                            return CastImpl<std::uint8_t>{}(temp);
                        } else if constexpr (sizeof(To) == 2) {
                            if constexpr (Saturating) {
                                constexpr auto fn = [](auto const& v_) {
                                    return cast_helper<To>(v_, [](auto m) {
                                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                        return _mm_packus_epi32(m, m);
                                        #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                        return _mm_packus_epi32(m, m);
                                        #else
                                        return _mm_packus_epi32(m, m);
                                        #endif
                                    });
                                };
                                return cast_iter_chunk<To>(
                                     v,
                                     Matcher {
                                        case_maker<4> = [fn](auto const& v_) {
                                            return fn(join(v_, v_));
                                        },
                                        case_maker<8> = fn
                                    }
                                 );
                            }
                        } else if (sizeof(To) == 4 && Saturating) {
                            return saturating_cast(v);
                        }
                    }
                    auto temp = CastImpl<std::make_signed_t<To>, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                }
            }


         }; 
    } // namespace internal

    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto cast(Vec<N, From> const& v) noexcept {
        return internal::CastImpl<To, false>{}(v);
    }

    template <typename To, std::size_t N, std::integral From>
    UI_ALWAYS_INLINE auto sat_cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return internal::CastImpl<To, true>{}(v);
    }

    // retinterpret cast
    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE constexpr auto rcast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return std::bit_cast<Vec<N, To>>(v);
    }
} // namespace ui:x86

#endif // AMT_UI_ARC_X86_HPP

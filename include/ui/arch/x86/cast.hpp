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
                static_assert(
                    sizeof(T) == sizeof(float)   ||
                    sizeof(T) == sizeof(double)  ||
                    sizeof(T) == sizeof(float16) || 
                    sizeof(T) == sizeof(bfloat16),
                    "Unknow floating-point type, expecting 'float', 'ui::float16', 'ui::bfloat16' or 'double'"
                );
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

    template <std::same_as<double> T>
    UI_ALWAYS_INLINE constexpr auto from_vec(__m128d v) noexcept -> Vec<2, double> {
        return std::bit_cast<Vec<2, double>>(v); 
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(__m128d v) noexcept -> Vec<2, double> {
        return std::bit_cast<Vec<2, double>>(v); 
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
            constexpr Matcher(Cs&&... cs) noexcept requires (sizeof...(Cs) > 1)
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

        template <typename To, bool Saturating, typename M, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto cast_iter_chunk(Vec<N, T> const& v, M const& m) noexcept -> Vec<N, To> {
            if constexpr (N == 1) {
                if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return Vec<N, To>{ .val = static_cast<To>(float(v.val)) };
                } else if constexpr (std::floating_point<To>) {
                    return Vec<N, To>{ .val = static_cast<To>(v.val) };
                } else {
                    return { .val = ::ui::internal::saturating_cast_helper<To, Saturating>(v.val) };
                }
            } else {
                if constexpr (is_case_invocable<N, M, decltype(v)>) {
                    auto temp = m.template match<N>(v);
                    if constexpr (::ui::internal::is_vec<decltype(temp)>) return temp;
                    else return from_vec<To>(temp);
                } else {
                    return join(cast_iter_chunk<To, Saturating>(v.lo, m), cast_iter_chunk<To, Saturating>(v.hi, m));
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
                static constexpr auto ratio = (sizeof(ret_t) / sizeof(To)) * (sizeof(T) / sizeof(std::uint8_t));

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

        template <typename To, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto clamp_lower_range_to_zero(Vec<N, T> const& v) noexcept -> Vec<N, T> {
            auto m = to_vec(v);
            if constexpr (sizeof(T) == 1) {
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    return from_vec<T>(_mm_max_epi8(_mm_setzero_si128(), m));
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    return from_vec<T>(_mm256_max_epi8(_mm256_setzero_si256(), m));
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    return from_vec<T>(_mm512_max_epi8(_mm512_setzero_si512(), m));
                }
                #endif
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    return from_vec<T>(_mm_max_epi16(_mm_setzero_si128(), m));
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    return from_vec<T>(_mm256_max_epi16(_mm256_setzero_si256(), m));
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    return from_vec<T>(_mm512_max_epi16(_mm512_setzero_si512(), m));
                }
                #endif
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    return from_vec<T>(_mm_max_epi32(_mm_setzero_si128(), m));
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    return from_vec<T>(_mm256_max_epi32(_mm256_setzero_si256(), m));
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    return from_vec<T>(_mm512_max_epi32(_mm512_setzero_si512(), m));
                }
                #endif
            } else if constexpr (sizeof(T) == 8) {
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    return from_vec<T>(_mm_max_epi64(_mm_setzero_si128(), m));
                }
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    return from_vec<T>(_mm256_max_epi64(_mm256_setzero_si256(), m));
                }
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    return from_vec<T>(_mm512_max_epi64(_mm512_setzero_si512(), m));
                }
                #else
                return map([](auto v_) {
                    return std::max<T>(v_, 0);
                }, v);;
                #endif
            }
        }

        template <typename To, std::size_t N, typename T>
        UI_ALWAYS_INLINE auto clamp_upper_range_to_max(Vec<N, T> const& v) noexcept -> Vec<N, T> {
            auto m = to_vec(v);
            static constexpr auto max = static_cast<T>(
                std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(std::numeric_limits<To>::max()),
                    static_cast<std::uint64_t>(std::numeric_limits<T>::max())
                )
            );
            #define CAST(PREFIX, BIT, VAL, MAX) \
                if (std::is_signed_v<T>) return from_vec<T>(PREFIX##_min_epi##BIT(VAL, MAX)); \
                else return from_vec<T>(PREFIX##_min_epu##BIT(VAL, MAX)); \

            if constexpr (sizeof(T) == 1) {
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    auto mx = _mm_set1_epi8(max);
                    CAST(_mm, 8, mx, m)
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    auto mx = _mm256_set1_epi8(max);
                    CAST(_mm256, 8, mx, m)
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    auto mx = _mm512_set1_epi8(max);
                    CAST(_mm512, 8, mx, m)
                }
                #endif
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    auto mx = _mm_set1_epi16(max);
                    CAST(_mm, 16, mx, m)
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    auto mx = _mm256_set1_epi16(max);
                    CAST(_mm256, 16, mx, m)
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    auto mx = _mm512_set1_epi8(max);
                    CAST(_mm512, 16, mx, m)
                }
                #endif
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    auto mx = _mm_set1_epi32(max);
                    CAST(_mm, 32, mx, m)
                }
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    auto mx = _mm256_set1_epi32(max);
                    CAST(_mm256, 32, mx, m)
                }
                #endif
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    auto mx = _mm512_set1_epi8(max);
                    CAST(_mm512, 32, mx, m)
                }
                #endif
            } else if constexpr (sizeof(T) == 8) {
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m128i)) {
                    auto mx = _mm_set1_epi64(max);
                    CAST(_mm, 64, mx, m)
                }
                if constexpr (sizeof(m) == sizeof(__m256i)) {
                    auto mx = _mm256_set1_epi64(max);
                    CAST(_mm256, 64, mx, m)
                }
                if constexpr (sizeof(m) == sizeof(__m512i)) {
                    auto mx = _mm512_set1_epi8(max);
                    CAST(_mm512, 64, mx, m)
                }
                #else
                return map([](auto v_) {
                    return std::min<T>(v_, static_cast<T>(std::numeric_limits<To>::max()));
                }, v);;
                #endif
            }
            #undef CAST
        }

        template <std::integral To, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto clamp_range(Vec<N, T> const& v) noexcept -> Vec<N, T> {
            auto m = to_vec(v);
            static constexpr auto max = static_cast<T>(
                std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(std::numeric_limits<To>::max()),
                    static_cast<std::uint64_t>(std::numeric_limits<T>::max())
                )
            );
            static constexpr auto min = static_cast<T>(
                std::max<std::int64_t>(
                    static_cast<std::int64_t>(std::numeric_limits<To>::min()),
                    static_cast<std::int64_t>(std::numeric_limits<T>::min())
                )
            );
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi8(max);
                        auto mi = _mm_set1_epi8(min); 
                        return from_vec<T>(
                            _mm_max_epi8(mi, _mm_min_epi8(mx, m))
                        );
                    }
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi8(max);
                        auto mi = _mm256_set1_epi8(min); 
                        return from_vec<T>(
                            _mm256_max_epi8(mi, _mm256_min_epi8(mx, m))
                        );
                    }
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi8(max);
                        auto mi = _mm512_set1_epi8(min); 
                        return from_vec<T>(
                            _mm512_max_epi8(mi, _mm512_min_epi8(mx, m))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi16(max);
                        auto mi = _mm_set1_epi16(min); 
                        return from_vec<T>(
                            _mm_max_epi16(mi, _mm_min_epi16(mx, m))
                        );
                    }
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi16(max);
                        auto mi = _mm256_set1_epi16(min); 
                        return from_vec<T>(
                            _mm256_max_epi16(mi, _mm256_min_epi16(mx, m))
                        );
                    }
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi16(max);
                        auto mi = _mm512_set1_epi16(min); 
                        return from_vec<T>(
                            _mm512_max_epi16(mi, _mm512_min_epi16(mx, m))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi32(max);
                        auto mi = _mm_set1_epi32(min); 
                        return from_vec<T>(
                            _mm_max_epi32(mi, _mm_min_epi32(mx, m))
                        );
                    }
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi32(max);
                        auto mi = _mm256_set1_epi32(min); 
                        return from_vec<T>(
                            _mm256_max_epi32(mi, _mm256_min_epi32(mx, m))
                        );
                    }
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi32(max);
                        auto mi = _mm512_set1_epi32(min); 
                        return from_vec<T>(
                            _mm512_max_epi32(mi, _mm512_min_epi32(mx, m))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi64(max);
                        auto mi = _mm_set1_epi64(min); 
                        return from_vec<T>(
                            _mm_max_epi64(mi, _mm_min_epi64(mx, m))
                        );
                    }
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi64(max);
                        auto mi = _mm256_set1_epi64(min); 
                        return from_vec<T>(
                            _mm256_max_epi64(mi, _mm256_min_epi64(mx, m))
                        );
                    }
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi64(max);
                        auto mi = _mm512_set1_epi64(min); 
                        return from_vec<T>(
                            _mm512_max_epi64(mi, _mm512_min_epi64(mx, m))
                        );
                    }
                    #else
                    return map([](auto v_) {
                        return std::clamp<T>(
                            v_,
                            min,
                            max
                        );
                    }, v);;
                    #endif
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi8(max);
                        auto mi = _mm_setzero_si128(); 
                        return from_vec<T>(
                            _mm_max_epu8(mi, _mm_min_epu8(mx, m))
                        );
                    }
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi8(max);
                        auto mi = _mm256_setzero_si256(); 
                        return from_vec<T>(
                            _mm256_max_epu8(mi, _mm256_min_epu8(mx, m))
                        );
                    }
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi8(max);
                        auto mi = _mm512_setzero_si512();
                        return from_vec<T>(
                            _mm512_max_epu8(mi, _mm512_min_epu8(mx, m))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi16(max);
                        auto mi = _mm_setzero_si128(); 
                        return from_vec<T>(
                            _mm_max_epu16(mi, _mm_min_epu16(mx, m))
                        );
                    }
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi16(max);
                        auto mi = _mm256_setzero_si256();
                        return from_vec<T>(
                            _mm256_max_epu16(mi, _mm256_min_epu16(mx, m))
                        );
                    }
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi16(max);
                        auto mi = _mm512_setzero_si512();
                        return from_vec<T>(
                            _mm512_max_epu16(mi, _mm512_min_epu16(mx, m))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi32(max);
                        auto mi = _mm_setzero_si128(); 
                        return from_vec<T>(
                            _mm_max_epu32(mi, _mm_min_epu32(mx, m))
                        );
                    }
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi32(max);
                        auto mi = _mm256_setzero_si256();
                        auto temp = from_vec<T>(
                            _mm256_max_epu32(mi, _mm256_min_epu32(mx, m))
                        );
                        return temp;
                    }
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi32(max);
                        auto mi = _mm512_setzero_si512();
                        return from_vec<T>(
                            _mm512_max_epu32(mi, _mm512_min_epu32(mx, m))
                        );
                    }
                    #endif
                } else if constexpr (sizeof(T) == 8) {
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    if constexpr (sizeof(m) == sizeof(__m128i)) {
                        auto mx = _mm_set1_epi64(max);
                        auto mi = _mm_setzero_si128(); 
                        return from_vec<T>(
                            _mm_max_epu64(mi, _mm_min_epu64(mx, m))
                        );
                    }
                    if constexpr (sizeof(m) == sizeof(__m256i)) {
                        auto mx = _mm256_set1_epi64(max);
                        auto mi = _mm256_setzero_si256();
                        return from_vec<T>(
                            _mm256_max_epu64(mi, _mm256_min_epu64(mx, m))
                        );
                    }
                    if constexpr (sizeof(m) == sizeof(__m512i)) {
                        auto mx = _mm512_set1_epi64(max);
                        auto mi = _mm512_setzero_si512();
                        return from_vec<T>(
                            _mm512_max_epu64(mi, _mm512_min_epu64(mx, m))
                        );
                    }
                    #else
                    return map([](auto v_) {
                        return std::clamp<T>(
                            v_,
                            0,
                            max
                        );
                    }, v);;
                    #endif
                }
            }
        }

        template <typename To, std::size_t N, std::integral T>
        UI_ALWAYS_INLINE auto saturating_helper(Vec<N, T> const& v) noexcept -> Vec<N, T> {
            if constexpr (std::is_signed_v<T> == std::is_signed_v<To>) {
                if constexpr (sizeof(T) <= sizeof(To)) {
                    return v;
                }
            }
            constexpr auto fn = [](auto const& v_) {
                /*if constexpr (sizeof(To) < sizeof(T)) return clamp_range<To>(v_);*/
                if constexpr (std::is_signed_v<T> == std::is_signed_v<To>) {
                    if constexpr (sizeof(To) < sizeof(T)) return clamp_range<To>(v_);
                    return v_;
                }
                if constexpr (std::is_signed_v<To>) {
                    return clamp_upper_range_to_max<To>(v_);
                } else {
                    return clamp_lower_range_to_zero<To>(v_);
                }
            };
            return cast_iter_chunk<T, true>(
                 v,
                 Matcher {
                    case_maker<128 / (sizeof(T) * 8 * 2)> = [fn](auto const& v_) {
                        return fn(join(v_, v_));
                    },
                    case_maker<128 / (sizeof(T) * 8)> = fn
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                    , case_maker<256 / (sizeof(T) * 8)> = fn
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    , case_maker<512 / (sizeof(T) * 8)> = fn
                    #endif
                }
             );
        }

        template <typename To, std::size_t N, std::floating_point T>
        UI_ALWAYS_INLINE auto apply_infinity_mask(Vec<N, T> const& v, Vec<N, To> const& t) noexcept -> Vec<N, To> {
            if constexpr (std::floating_point<To>) return t;
            return map([](auto v_, auto t_) {
                return v_ == std::numeric_limits<T>::infinity()
                    ? std::numeric_limits<To>::max()
                    : (v_ == -std::numeric_limits<T>::infinity() ? std::numeric_limits<To>::min() : t_);
            }, v, t);
        }

        template <std::size_t N, typename T>
            requires (!std::is_signed_v<T> && !std::is_void_v<decltype(to_vec(Vec<N, T>{}))>)
        UI_ALWAYS_INLINE auto convert_unsigned_to_float_helper(
            Vec<N, T> const& v
        ) noexcept -> Vec<N, float> {
            auto m = to_vec(v);
            #define SHIFT(P, V)         _##P##_srli_epi32(V, sizeof(T) * 8 - 1)
            #define BROADCAST(P, V)     _##P##_set1_epi32(V)
            #define BROADCAST_F(P, V)   _##P##_set1_ps(V)
            #define AND(P, L, R, B)     _##P##_and_si##B(L, R)
            #define F(P, I)             _##P##_cvtepi32_ps(I)
            #define MUL(P, L, R)        _##P##_mul_ps(L, R)
            #define ADD(P, L, R)        _##P##_add_ps(L, R)

        #define OP(P, B) auto high = SHIFT(P, m);\
            static constexpr auto low_mask_val = std::numeric_limits<T>::max() >> 1;\
            auto low_mask = BROADCAST(P, low_mask_val);\
            auto low = AND(P, m, low_mask, B);\
            auto base = F(P, low);\
            auto high_f = F(P, high);\
            auto offset = MUL(P, high_f, BROADCAST_F(P, low_mask_val));\
            return from_vec<float>(ADD(P, base, offset));

            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX2
                OP(mm, 128)
            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m128)) {
                    OP(mm, 128)
                } else if constexpr (sizeof(m) == sizeof(__m256)) {
                    OP(mm, 256)
                }
            #else
                if constexpr (sizeof(m) == sizeof(__m128)) {
                    OP(mm, 128)
                } else if constexpr (sizeof(m) == sizeof(__m256)) {
                    OP(mm, 256)
                } else {
                    OP(mm, 512)
                }
            #endif

            #undef SHIFT
            #undef BROADCAST
            #undef BROADCAST_F
            #undef AND
            #undef F
            #undef MUL
            #undef ADD
            #undef OP
        }

        template <std::size_t N, typename T>
            requires (!std::is_signed_v<T>)
        UI_ALWAYS_INLINE auto convert_unsigned_to_double_helper(
            Vec<N, T> const& v
        ) noexcept {
            auto m = to_vec(v);
            #define SHIFT(P, V)         _##P##_srli_epi32(V, sizeof(T) * 8 - 1)
            #define BROADCAST(P, V)     _##P##_set1_epi32(V)
            #define BROADCAST_F(P, V)   _##P##_set1_pd(V)
            #define AND(P, L, R, B)     _##P##_and_si##B(L, R)
            #define F(P, I)             _##P##_cvtepi32_pd(I)
            #define MUL(P, L, R)        _##P##_mul_pd(L, R)
            #define ADD(P, L, R)        _##P##_add_pd(L, R)

        #define OP(P, B) auto high = SHIFT(P, m);\
            static constexpr auto low_mask_val = std::numeric_limits<T>::max() >> 1;\
            auto low_mask = BROADCAST(P, low_mask_val);\
            auto low = AND(P, m, low_mask, B);\
            auto base = F(P, low);\
            auto high_f = F(P, high);\
            auto offset = MUL(P, high_f, BROADCAST_F(P, low_mask_val));\
            return from_vec<double>(ADD(P, base, offset));

            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX2
                OP(mm, 128)
            #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                if constexpr (sizeof(m) == sizeof(__m128)) {
                    OP(mm, 128)
                } else if constexpr (sizeof(m) == sizeof(__m256)) {
                    OP(mm256, 256)
                }
            #else
                if constexpr (sizeof(m) == sizeof(__m128)) {
                    OP(mm, 128)
                } else if constexpr (sizeof(m) == sizeof(__m256)) {
                    OP(mm256, 256)
                } else {
                    OP(mm512, 512)
                }
            #endif

            #undef SHIFT
            #undef BROADCAST
            #undef BROADCAST_F
            #undef AND
            #undef F
            #undef MUL
            #undef ADD
            #undef OP
        }

        template <std::size_t N, typename T>
            requires (!std::is_signed_v<T> && sizeof(T) == 4)
        UI_ALWAYS_INLINE auto convert_unsigned_to_double(
            Vec<N, T> const& v
        ) noexcept -> Vec<N, double> {
            if constexpr (N == 1) {
                auto t = Vec<4, std::uint32_t>::load(v[0]);
                auto val = convert_unsigned_to_double_helper(t)[0];
                return { .val = val };
            } else if constexpr (!std::is_void_v<decltype(to_vec(v))>) {
                return join(
                    convert_unsigned_to_double_helper(join(v.lo, v.lo)),
                    convert_unsigned_to_double_helper(join(v.hi, v.hi))
                );
            } else {
                return join(
                    convert_unsigned_to_double(v.lo),
                    convert_unsigned_to_double(v.hi)
                );
            }
        }

        template <std::size_t N, typename T>
            requires (!std::is_signed_v<T> && sizeof(T) == 4)
        UI_ALWAYS_INLINE auto convert_unsigned_to_float(
            Vec<N, T> const& v
        ) noexcept -> Vec<N, float> {
            if constexpr (N == 1) {
                auto t = Vec<4, std::uint32_t>::load(v[0]);
                auto val = convert_unsigned_to_float_helper(t)[0];
                return { .val = val };
            } else if constexpr (!std::is_void_v<decltype(to_vec(v))>) {
                return convert_unsigned_to_float_helper(v);
            } else {
                return join(
                    convert_unsigned_to_float(v.lo),
                    convert_unsigned_to_float(v.hi)
                );
            }
        }

        template <typename To, bool Saturating = false, bool ClampFp = true>
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
                   return cast_iter_chunk<To, Saturating>(
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
                   return cast_iter_chunk<To, Saturating>(
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
                        return cast_iter_chunk<To, Saturating>(
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
                         return cast_iter_chunk<To, Saturating>(
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
                         return cast_iter_chunk<To, Saturating>(
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
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    auto temp = CastImpl<std::make_signed_t<To>>{}(vt);
                    return std::bit_cast<Vec<N, To>>(temp);
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
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    auto temp = CastImpl<std::make_unsigned_t<To>>{}(vt);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else {
                    if constexpr (sizeof(To) == 1) return v;
                    else if constexpr (sizeof(To) == 2) {
                         return cast_iter_chunk<To, Saturating>(
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
                        return cast_iter_chunk<To, Saturating>(
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
                        return cast_iter_chunk<To, Saturating>(
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
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return CastImpl<float>{}(temp);
                } else if constexpr (std::same_as<To, double>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
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
                        return cast_iter_chunk<To, Saturating>(
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
                                return _mm_cvtepi16_epi32(m);
                                #else
                                return _mm256_cvtepi16_epi32(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To, Saturating>(
                             v,
                             Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<8> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<16> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    return _mm512_cvtepi16_epi32(m); 
                                }
                                #endif
                            }
                         );
                    } else if constexpr (sizeof(To) == 8) {
                        constexpr auto fn = [](auto const& v_) {
                            return cast_helper<To>(v_, [](__m128i m) {
                                #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                return _mm_cvtepi16_epi64(m);
                                #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                return _mm256_cvtepi16_epi64(m);
                                #else
                                return _mm512_cvtepi16_epi64(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To, Saturating>(
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
                            return cast_iter_chunk<To, Saturating>(
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

                        }
                    }

                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    if constexpr (sizeof(To) == 2) return std::bit_cast<Vec<N, To>>(vt);
                    auto temp = CastImpl<std::make_signed_t<To>>{}(vt);
                    return std::bit_cast<Vec<N, To>>(temp);
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint16_t> const& v
            ) noexcept -> Vec<N, To> {
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
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    auto temp = CastImpl<std::make_signed_t<To>>{}(vt);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else {
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    if constexpr (sizeof(To) == 1) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (Saturating) {
                                return from_vec<To>(_mm_packus_epi16(m, m)).lo;
                            } else {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                auto temp = _mm_shuffle_epi8(m, *reinterpret_cast<__m128i const*>(constants::mask8_16_even_odd));
                                return from_vec<To>(temp).lo;
                            #else
                                return from_vec<To>(_mm_cvtepu16_epi8(m)).lo;
                            #endif
                            } 
                        };
                        return cast_iter_chunk<To, false>(
                            vt,
                            Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_,v_)).lo;
                                },
                                case_maker<8> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                                , case_maker<16> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        auto vec = from_vec<To>(_mm256_packus_epi16(m, m));
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
                                        return from_vec<To>(_mm256_cvtepu16_epi8(m));
                                    #endif
                                    } 
                                }
                                #endif
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<32> = [](auto const& v_) {
                                    auto m = to_vec(v_);
                                    if constexpr (Saturating) {
                                        return from_vec<To>(_mm512_packus_epi16(m, m)).lo;
                                    } else {
                                        return from_vec<To>(_mm512_cvtepu16_epi8(m));
                                    } 
                                }
                                #endif
                            }
                        );
                    } else if constexpr (sizeof(To) == 2) {
                        return std::bit_cast<Vec<N, To>>(vt);
                    } else if constexpr (sizeof(To) == 4) {
                        constexpr auto fn = [](auto const& v_) {
                            return cast_helper<To>(v_, [](__m128i m) {
                                #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                return _mm_cvtepu16_epi32(m);
                                #else
                                return _mm256_cvtepu16_epi32(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To, false>(
                             vt,
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
                        return cast_iter_chunk<To, false>(
                             vt,
                             Matcher {
                                case_maker<4> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<8> = fn
                            }
                         );
                    }
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
                            if constexpr (sizeof(m) == sizeof(__m128)) {
                                return _mm_cvtepi32_ps(m);
                            }
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            if constexpr (sizeof(m) == sizeof(__m256)) {
                                return _mm256_cvtepi32_ps(m);
                            }
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            if constexpr (sizeof(m) == sizeof(__m512)) {
                                return _mm512_cvtepi32_ps(m);
                            }
                            #endif
                        });
                    };
                    return cast_iter_chunk<To, Saturating>(
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
                    return cast_iter_chunk<To, Saturating>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<8> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::is_signed_v<To>) {
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(v); 
                    }
                    if constexpr (sizeof(To) == 1) {
                        using type = std::conditional_t<
                            std::is_signed_v<To>,
                            std::int16_t,
                            std::uint16_t
                        >;
                        auto temp = CastImpl<type>{}(vt);
                        return CastImpl<To>{}(temp);
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
                        return cast_iter_chunk<To, false>(
                            vt,
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
                        return cast_iter_chunk<To, Saturating>(
                             v,
                             Matcher {
                                case_maker<2> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<4> = fn
                            }
                         );
                    }
                } else {
                    if constexpr (Saturating) {
                        if constexpr (sizeof(To) == 2) {
                            constexpr auto fn = [](auto const& v_) {
                                auto m = to_vec(v_);
                                if constexpr (sizeof(m) == sizeof(__m128)) {
                                    return from_vec<To>(_mm_packus_epi32(m, m)).lo;
                                }
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                                if constexpr (sizeof(m) == sizeof(__m256)) {
                                    return from_vec<To>(_mm256_packus_epi32(m, m)).lo;
                                }
                                #elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                if constexpr (sizeof(m) == sizeof(__m256)) {
                                    return from_vec<To>(_mm512_packus_epi32(m, m)).lo;
                                }
                                #endif
                            };
                            return cast_iter_chunk<To, Saturating>(
                                 v,
                                 Matcher {
                                    case_maker<2> = [fn](auto const& v_) {
                                        return fn(join(v_, v_));
                                    },
                                    case_maker<4> = fn
                                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX2
                                    , case_maker<8> = fn
                                    #endif
                                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                    , case_maker<16> = fn
                                    #endif
                                }
                             );
                        }
                    }
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    if constexpr (sizeof(To) == 4) return std::bit_cast<Vec<N, To>>(vt);
                    return std::bit_cast<Vec<N, To>>(CastImpl<std::make_signed_t<To>>{}(vt));
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint32_t> const& v
            ) noexcept -> Vec<N, To> {
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
                    return cast_iter_chunk<To, Saturating>(
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
                    return cast_iter_chunk<To, Saturating>(
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
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(v);
                    }
                    if constexpr (sizeof(To) == 1) {
                        using type = std::conditional_t<
                            std::is_signed_v<To>,
                            std::int16_t,
                            std::uint16_t
                        >;
                        auto temp = CastImpl<type>{}(vt);
                        return CastImpl<To>{}(temp);
                    } else if constexpr (sizeof(To) == 2) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (Saturating) {
                                return from_vec<To>(_mm_packus_epi32(m, m)).lo;
                            } else {
                            #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                auto temp = _mm_shuffle_epi8(m, *reinterpret_cast<__m128i const*>(constants::mask8_32_even_odd));
                                return from_vec<To>(temp).lo;
                            #else
                                return from_vec<To>(_mm_cvtepu32_epi16(m)).lo;
                            #endif
                            } 
                        };
                        return cast_iter_chunk<To, Saturating>(
                            vt,
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
                                        return from_vec<To>(_mm256_cvtepu32_epi16(m));
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
                                        return from_vec<To>(_mm512_cvtepu32_epi16(m));
                                    } 
                                }
                                #endif
                            }
                        );
                    } else if constexpr (sizeof(To) == 4) {
                        return std::bit_cast<Vec<N, To>>(vt);
                    } else if constexpr (sizeof(To) == 8) {
                        constexpr auto fn = [](auto const& v_) {
                            return cast_helper<To>(v_, [](__m128i m) {
                                #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                                return _mm_cvtepu32_epi64(m);
                                #elif UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_SKX
                                return _mm256_cvtepu32_epi64(m);
                                #else
                                return _mm512_cvtepu32_epi64(m);
                                #endif
                            });
                        };
                        return cast_iter_chunk<To, Saturating>(
                             vt,
                             Matcher {
                                case_maker<2> = [fn](auto const& v_) {
                                    return fn(join(v_, v_));
                                },
                                case_maker<4> = fn
                            }
                         );
                    }
                } else {
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(v);
                    }
                    if constexpr (sizeof(To) == 1) {
                        using type = std::conditional_t<
                            std::is_signed_v<To>,
                            std::int16_t,
                            std::uint16_t
                        >;
                        auto temp = CastImpl<type>{}(vt);
                        return CastImpl<To>{}(temp);
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
                        return cast_iter_chunk<To, false>(
                            vt,
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
                        return cast_iter_chunk<To, Saturating>(
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
                    [[maybe_unused]] constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            using type = std::decay_t<decltype(m)>;
                            if constexpr (sizeof(type) * 8 == 128) {
                                return from_vec<To>(_mm_cvtepi64_ps(m)).lo;
                            } else if constexpr (sizeof(type) * 8 == 256) {
                                return from_vec<To>(_mm256_cvtepi64_ps(m));
                            } else {
                                return from_vec<To>(_mm512_cvtepi64_ps(m));
                            }
                            #endif
                        });
                    };
                    return cast_iter_chunk<To, Saturating>(
                         v,
                         Matcher {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            case_maker<1> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<2> = fn,
                            case_maker<4> = fn,
                            case_maker<8> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::same_as<To, double>) {
                    [[maybe_unused]] constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            using type = std::decay_t<decltype(m)>;
                            if constexpr (sizeof(type) * 8 == 128) {
                                return from_vec<To>(_mm_cvtepi64_pd(m)).lo;
                            } else if constexpr (sizeof(type) * 8 == 256) {
                                return from_vec<To>(_mm256_cvtepi64_pd(m));
                            } else {
                                return from_vec<To>(_mm512_cvtepi64_ps(m));
                            }
                            #endif
                        });
                    };
                    return cast_iter_chunk<To, Saturating>(
                         v,
                         Matcher {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            case_maker<1> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<2> = fn,
                            case_maker<4> = fn,
                            case_maker<8> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 8) return v;
                    else {
                        auto vt = v;
                        if constexpr (Saturating) {
                            vt = saturating_helper<To>(vt);
                        }
                        return cast_iter_chunk<To, false>(
                             vt,
                             Matcher {}
                        );
                    }
                } else {
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    if constexpr (sizeof(To) == 8) {
                        if constexpr (sizeof(To) == 8) return std::bit_cast<Vec<N, To>>(vt);
                        return std::bit_cast<Vec<N, To>>(vt);
                    } else {
                        return cast_iter_chunk<To, false>(
                             vt,
                             Matcher {}
                        );
                    }
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::uint64_t> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                   auto temp = CastImpl<float, Saturating>{}(v);
                   return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, float>) {
                    [[maybe_unused]] constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            using type = std::decay_t<decltype(m)>;
                            if constexpr (sizeof(type) * 8 == 128) {
                                return from_vec<To>(_mm_cvtepu64_ps(m)).lo;
                            } else if constexpr (sizeof(type) * 8 == 256) {
                                return from_vec<To>(_mm256_cvtepu64_ps(m));
                            } else {
                                return from_vec<To>(_mm512_cvtepu64_ps(m));
                            }
                            #endif
                        });
                    };
                    return cast_iter_chunk<To, Saturating>(
                         v,
                         Matcher {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            case_maker<1> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<2> = fn,
                            case_maker<4> = fn,
                            case_maker<8> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::same_as<To, double>) {
                    [[maybe_unused]] constexpr auto fn = [](auto const& v_) {
                        return cast_helper<To>(v_, [](auto m) {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            using type = std::decay_t<decltype(m)>;
                            if constexpr (sizeof(type) * 8 == 128) {
                                return from_vec<To>(_mm_cvtepu64_pd(m)).lo;
                            } else if constexpr (sizeof(type) * 8 == 256) {
                                return from_vec<To>(_mm256_cvtepu64_pd(m));
                            } else {
                                return from_vec<To>(_mm512_cvtepu64_ps(m));
                            }
                            #endif
                        });
                    };
                    return cast_iter_chunk<To, Saturating>(
                         v,
                         Matcher {
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            case_maker<1> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<2> = fn,
                            case_maker<4> = fn,
                            case_maker<8> = fn
                            #endif
                        }
                     );
                } else if constexpr (!std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 8) return v;
                    else {
                        auto vt = v;
                        if constexpr (Saturating) {
                            vt = saturating_helper<To>(vt);
                        }
                        return cast_iter_chunk<To, false>(
                             vt,
                             Matcher {}
                        );
                    }
                } else {
                    auto vt = v;
                    if constexpr (Saturating) {
                        vt = saturating_helper<To>(vt);
                    }
                    if constexpr (sizeof(To) == 8) {
                        return std::bit_cast<Vec<N, To>>(vt);
                    } else {
                        return cast_iter_chunk<To, false>(
                             vt,
                             Matcher {}
                        );
                    }
                }
            }


            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, float> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                    return cast_float32_to_float16(v);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    return cast_float32_to_bfloat16(v);
                } else if constexpr (std::same_as<To, float>) {
                    return v;
                } else if constexpr (std::same_as<To, double>) {
                    constexpr auto fn = [](auto const& v_) {
                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                        if constexpr (sizeof(v_) == sizeof(__m128)) {
                            auto t0 = join(v_.lo, v_.lo);
                            auto t1 = join(v_.hi, v_.hi);
                            return join(
                                from_vec<To>(_mm_cvtps_pd(to_vec(t0))).lo,
                                from_vec<To>(_mm_cvtps_pd(to_vec(t1))).lo
                            );
                        }
                        #else
                        if constexpr (sizeof(v_) == sizeof(__m128)) {
                            return from_vec<To>(_mm256_cvtps_pd(to_vec(v_)));
                        }
                        #endif
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        if constexpr (sizeof(v_) == sizeof(__m256)) {
                            return from_vec<To>(_mm512_cvtps_pd(to_vec(v_)));
                        }
                        #endif
                    };
                    return cast_iter_chunk<To, Saturating>(
                         v,
                         Matcher {
                            case_maker<2> = [fn](auto const& v_) {
                                return fn(join(v_, v_));
                            },
                            case_maker<4> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<8> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 4) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (sizeof(m) == sizeof(__m128)) {
                                return from_vec<To>(_mm_cvtps_epi32(m));
                            }
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            if constexpr (sizeof(m) == sizeof(__m256)) {
                                return from_vec<To>(_mm256_cvtps_epi32(m));
                            }
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            if constexpr (sizeof(m) == sizeof(__m256)) {
                                return from_vec<To>(_mm512_cvtps_epi32(m));
                            }
                            #endif
                        };
                        auto temp = cast_iter_chunk<To, Saturating>(
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
                        if constexpr (ClampFp) {
                            return apply_infinity_mask(v, temp);
                        } else {
                            return temp;
                        }
                    }
                }

                auto t0 = CastImpl<std::int32_t, Saturating, false>{}(v);
                auto t1 = CastImpl<To, Saturating, false>{}(t0);
                if constexpr (ClampFp) {
                    return apply_infinity_mask(v, t1);
                } else {
                    return t1;
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, double> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                    auto temp = CastImpl<float>{}(v);
                    return cast_float32_to_float16(temp);
                } else if constexpr (std::same_as<To, bfloat16>) {
                    auto temp = CastImpl<float>{}(v);
                    return cast_float32_to_bfloat16(temp);
                } else if constexpr (std::same_as<To, double>) {
                    return v;
                } else if constexpr (std::same_as<To, float>) {
                    constexpr auto fn = [](auto const& v_) {
                        auto m = to_vec(v_);
                        #if UI_CPU_SSE_LEVEL < UI_CPU_SSE_LEVEL_AVX
                        if constexpr (sizeof(v_) == sizeof(__m128)) {
                            return from_vec<To>(_mm_cvtpd_ps(m)).lo;
                        }
                        #else
                        if constexpr (sizeof(v_) == sizeof(__m128)) {
                            return from_vec<To>(_mm256_cvtpd_ps(m));
                        }
                        #endif
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        if constexpr (sizeof(v_) == sizeof(__m256)) {
                            return from_vec<To>(_mm512_cvtpd_ps(m));
                        }
                        #endif
                    };
                    return cast_iter_chunk<To, Saturating>(
                         v,
                         Matcher {
                            case_maker<1> = [fn](auto const& v_) {
                                return Vec<1, To>{ .val = static_cast<To>(v_.val) };
                            },
                            case_maker<2> = fn
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            , case_maker<4> = fn
                            , case_maker8> = fn
                            #endif
                        }
                     );
                } else if constexpr (std::is_signed_v<To>) {
                    if constexpr (sizeof(To) == 4) {
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (sizeof(m) == sizeof(__m128)) {
                                return from_vec<To>(_mm_cvtpd_epi32(m));
                            }
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                            if constexpr (sizeof(m) == sizeof(__m256)) {
                                return from_vec<To>(_mm256_cvtpd_epi32(m));
                            }
                            #endif
                            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                            if constexpr (sizeof(m) == sizeof(__m256)) {
                                return from_vec<To>(_mm512_cvtpd_epi32(m));
                            }
                            #endif
                        };
                        auto temp = cast_iter_chunk<To, Saturating>(
                             v,
                             Matcher {
                                case_maker<1> = [fn](auto const& v_) {
                                    return Vec<1, To>{ .val = static_cast<To>(v_.val) };
                                },
                                case_maker<2> = fn
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                                , case_maker<4> = fn
                                #endif
                                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                                , case_maker<8> = fn
                                #endif
                            }
                         );
                        if constexpr (ClampFp) {
                            return apply_infinity_mask(v, temp);
                        } else {
                            return temp;
                        }
                    } else if constexpr (sizeof(To) == 8) {
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        constexpr auto fn = [](auto const& v_) {
                            auto m = to_vec(v_);
                            if constexpr (sizeof(m) == sizeof(__m128)) {
                                return from_vec<To>(_mm_cvtpd_epi64(m));
                            }
                            if constexpr (sizeof(m) == sizeof(__m256)) {
                                return from_vec<To>(_mm256_cvtpd_epi64(m));
                            }
                            if constexpr (sizeof(m) == sizeof(__m256)) {
                                return from_vec<To>(_mm512_cvtpd_epi64(m));
                            }
                        };
                        auto temp = cast_iter_chunk<To, Saturating>(
                             v,
                             Matcher {
                                case_maker<1> = [fn](auto const& v_) {
                                    return Vec<1, To>{ .val = static_cast<To>(v_.val) };
                                },
                                case_maker<2> = fn,
                                case_maker<4> = fn,
                                case_maker<8> = fn
                            }
                         );
                        if constexpr (ClampFp) {
                            return apply_infinity_mask(v, temp);
                        } else {
                            return temp;
                        }
                        #else
                        return map([](auto v_) {
                            if constexpr (ClampFp) {
                                if (v_ == INFINITY) {
                                    return std::numeric_limits<To>::max();
                                } else if (v_ == -INFINITY) {
                                    return std::numeric_limits<To>::min();
                                }
                            }
                            return static_cast<To>(v_);
                        }, v);
                        #endif
                    }
                }

                auto t0 = CastImpl<std::int64_t, Saturating, false>{}(v);
                auto t1 = CastImpl<To, Saturating, false>{}(t0);
                if constexpr (ClampFp) {
                    return apply_infinity_mask(v, t1);
                } else {
                    return t1;
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, float16> const& v
            ) noexcept {
                if constexpr (std::same_as<To, float16>) {
                    return v;
                } else if constexpr (std::same_as<To, float>) {
                    return cast_float16_to_float32(v);
                } else {
                    auto temp = CastImpl<float>{}(v);
                    return CastImpl<To>{}(temp);
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, bfloat16> const& v
            ) noexcept {
                if constexpr (std::same_as<To, bfloat16>) {
                    return v;
                } else if constexpr (std::same_as<To, float>) {
                    return cast_bfloat16_to_float32(v);
                } else {
                    auto temp = CastImpl<float>{}(v);
                    return CastImpl<To>{}(temp);
                }
            }
         }; 
    } // namespace internal

    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
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

    template <std::size_t R, std::size_t C, typename T>
    UI_ALWAYS_INLINE constexpr auto to_vec(VecMat<R, C, T> const& m) noexcept {
        if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16> || std::integral<T>) {
            if constexpr (sizeof(m) == sizeof(__m128)) {
                return std::bit_cast<__m128i>(m); // 16bit integers for (b)float16
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (sizeof(m) == sizeof(__m256)) {
                return std::bit_cast<__m256i>(m);
            }
            #endif
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (sizeof(m) == sizeof(__m256)) {
                return std::bit_cast<__m512i>(m);
            }
            #endif
        } else if constexpr (std::same_as<T, float>) {
            if constexpr (sizeof(m) == sizeof(__m128)) {
                return std::bit_cast<__m128>(m);
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (sizeof(m) == sizeof(__m256)) {
                return std::bit_cast<__m256>(m);
            }
            #endif
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (sizeof(m) == sizeof(__m256)) {
                return std::bit_cast<__m512>(m);
            }
            #endif
        } else if constexpr (std::same_as<T, double>) {
            if constexpr (sizeof(m) == sizeof(__m128)) {
                return std::bit_cast<__m128d>(m);
            }
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
            if constexpr (sizeof(m) == sizeof(__m256)) {
                return std::bit_cast<__m256d>(m);
            }
            #endif
            #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
            if constexpr (sizeof(m) == sizeof(__m256)) {
                return std::bit_cast<__m512d>(m);
            }
            #endif
        }
    }

    template <unsigned R, unsigned C, typename T>
    UI_ALWAYS_INLINE constexpr auto from_vec(T const& v) noexcept {
        if constexpr (sizeof(T) * R * C == sizeof(__m128)) {
            return std::bit_cast<VecMat<R, C, T>>(v);
        }
        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
        if constexpr (sizeof(T) * R * C == sizeof(__m256)) {
            return std::bit_cast<VecMat<R, C, T>>(v);
        }
        #endif
        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
        if constexpr (sizeof(T) * R * C == sizeof(__m256)) {
            return std::bit_cast<VecMat<R, C, T>>(v);
        }
        #endif
    }
} // namespace ui:x86

#endif // AMT_UI_ARC_X86_HPP

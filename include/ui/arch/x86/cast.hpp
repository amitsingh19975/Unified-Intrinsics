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
                #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE2
                return std::bit_cast<__m128i>(v);
                #endif
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
         template <typename... Ts>
         struct Overloaded: Ts... {
            using Ts::operator()...;
         };
         
         template<typename... Ts>
         Overloaded(Ts...) -> Overloaded<Ts...>;

         template <typename To, typename Fn, std::size_t N, typename T>
         UI_ALWAYS_INLINE auto cast_helper(
            Vec<N, T> const& v,
            Fn&& fn
         ) noexcept {
            if constexpr (N == 1) {
               if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                  return Vec<N, To>{ .val = static_cast<To>(float(v.val)) };
               } else {
                  return Vec<N, To>{ .val = static_cast<To>(v.val) };
               }
            } else {
                static constexpr auto size = N * sizeof(T);
                if constexpr (
                    sizeof(T) < sizeof(To) && 
                    (size * 2 == sizeof(__m128)) && 
                    std::invocable<Fn, decltype(to_vec(Vec<2 * N, T>{}))>)
                {
                    Vec<2 * N, T> vt(v, v);
                    return from_vec<To>(fn(to_vec(vt)));
                } else {
                    if constexpr (size == sizeof(__m128) && std::invocable<Fn, decltype(to_vec(v))>) {
                        // widening cast
                        if constexpr (sizeof(T) < sizeof(To)) {
                            return join(
                                from_vec<To>(fn(to_vec(Vec<N, T>(v.lo, v.lo)))),
                                from_vec<To>(fn(to_vec(Vec<N, T>(v.hi, v.hi))))
                            );
                        } else {
                            return from_vec<To>(fn(to_vec(v)));
                        }
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                    } else if constexpr (size == sizeof(__m256) && std::invocable<Fn, decltype(to_vec(v))>) {
                         return from_vec<To>(fn(to_vec(v)));
                    #endif
                    #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                    } else if constexpr (size == sizeof(__m512) && std::invocable<Fn, decltype(to_vec(v)>) {
                         return from_vec<To>(fn(to_vec(v))),
                    #endif
                    }
                    return join(
                       cast_helper<To>(v.lo, fn),
                       cast_helper<To>(v.hi, fn)
                    );
                }
            }
         }

         template <typename To, bool Saturating = false>
         struct CastImpl {
            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
               Vec<N, std::int8_t> const& v      
            ) noexcept -> Vec<N, To> {
               if constexpr (std::same_as<To, float16>) {
                  auto temp = CastImpl<float, Saturating>{}(v);
                  return cast_float32_to_float16(temp);
               } else if constexpr (std::same_as<To, bfloat16>) {
                  auto temp = CastImpl<float, Saturating>{}(v);
                  return cast_float32_to_bfloat16(temp);
               } else if constexpr (std::same_as<To, float>) {
                  auto temp = CastImpl<std::int32_t>{}(v);
                  return cast_helper<To>(
                     temp,
                     Overloaded {
                        [](__m128i data) { return _mm_cvtepi32_ps(data);  }
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                        , [](__m256i data) { return _mm256_cvtepi32_ps(data);  }
                        #endif
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        , [](__m512i data) { return _mm512_cvtepi32_ps(data);  }
                        #endif
                     }
                  );   
               } else if constexpr (std::same_as<To, double>) {
                  auto temp = CastImpl<std::int64_t>{}(v);
                  return cast_helper<To>(
                     temp,
                     Overloaded {
                        #if UI_CPU_SSE_LEVEL > UI_CPU_SSE_LEVEL_SSE41
                        [](__m128i data) { return _mm_cvtepi64_pd(data);  }
                        #endif
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                        , [](__m256i data) { return _mm256_cvtepi64_pd(data);  }
                        #endif
                        #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                        , [](__m512i data) { return _mm512_cvtepi64_pd(data);  }
                        #endif
                     }
                  );   
               } else if constexpr (std::is_signed_v<To>) {
                  if constexpr (sizeof(To) == 1) {
                     return v;
                  } else if constexpr (sizeof(To) == 2) {
                        return cast_helper<To>(
                           v,
                           Overloaded {
                              [](__m128i data) { return _mm_cvtepi8_epi16(data);  }
                              #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                              , [](__m256i data) { return _mm256_cvtepi8_epi16(data);  }
                              #endif
                              #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                              , [](__m512i data) { return _mm512_cvtepi8_epi16(data);  }
                              #endif
                           }
                        );   
                  } else if constexpr (sizeof(To) == 4) {
                        return cast_helper<To>(
                           v,
                           Overloaded {
                              [](__m128i data) { return _mm_cvtepi8_epi32(data);  }
                              #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                              , [](__m256i data) { return _mm256_cvtepi8_epi32(data);  }
                              #endif
                              #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                              , [](__m512i data) { return _mm512_cvtepi8_epi32(data);  }
                              #endif
                           }
                        );   
                  } else if constexpr (sizeof(To) == 4) {
                        return cast_helper<To>(
                           v,
                           Overloaded {
                              [](__m128i data) { return _mm_cvtepi8_epi64(data);  }
                              #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
                              , [](__m256i data) { return _mm256_cvtepi8_epi64(data);  }
                              #endif
                              #if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SKX
                              , [](__m512i data) { return _mm512_cvtepi8_epi64(data);  }
                              #endif
                           }
                        );   
                  }

               } else {
                  return std::bit_cast<Vec<N, To>>(CastImpl<std::make_signed_t<To>>{}(v));            
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
} // namespace ui:x86

#endif // AMT_UI_ARC_X86_HPP

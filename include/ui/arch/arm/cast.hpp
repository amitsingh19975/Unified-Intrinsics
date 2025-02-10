#ifndef AMT_UI_ARCH_ARM_CAST_HPP
#define AMT_UI_ARCH_ARM_CAST_HPP

#include "../../base_vec.hpp"
#include "../../base.hpp"
#include "../../vec_headers.hpp"
#include "../../float.hpp"
#include "../../matrix.hpp"
#include <arm_neon.h>
#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace ui::arm::neon {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto to_vec(Vec<N, T> const& v) noexcept {
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<float, T>) {
                if constexpr (N == 1) return std::bit_cast<float>(v);
                else if constexpr (N == 2) return std::bit_cast<float32x2_t>(v);
                else return std::bit_cast<float32x4_t>(v);
            } else if constexpr (std::same_as<double, T>) {
                if constexpr (N == 1) return std::bit_cast<float64x1_t>(v);
                else return std::bit_cast<float64x2_t>(v);
            } else if constexpr (std::same_as<T, ui::float16>) {
                if constexpr (N == 1) return std::bit_cast<ui::float16>(v);
                else if constexpr (N == 4) return std::bit_cast<float16x4_t>(v);
                else return std::bit_cast<float16x8_t>(v);
            } else {
                static_assert(sizeof(T) == sizeof(float) || sizeof(T) == sizeof(double), "Unknow floating-point type, expecting 'float', 'ui::float16_t' or 'double'");
            }
        } else if constexpr (std::is_signed_v<T>) {
            if constexpr (sizeof(T) == 1) {
                if constexpr (N == 1) return std::bit_cast<T>(v);
                else if constexpr (N == 8) return std::bit_cast<int8x8_t>(v);
                else return std::bit_cast<int8x16_t>(v);
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (N == 1) return std::bit_cast<T>(v);
                else if constexpr (N == 4) return std::bit_cast<int16x4_t>(v);
                else return std::bit_cast<int16x8_t>(v);
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 1) return std::bit_cast<T>(v);
                else if constexpr (N == 2) return std::bit_cast<int32x2_t>(v);
                else return std::bit_cast<int32x4_t>(v);
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 1) return std::bit_cast<int64x1_t>(v);
                else return std::bit_cast<int64x2_t>(v);
            }
        } else {
            if constexpr (sizeof(T) == 1) {
                if constexpr (N == 1) return std::bit_cast<T>(v);
                else if constexpr (N == 8) return std::bit_cast<uint8x8_t>(v);
                else return std::bit_cast<uint8x16_t>(v);
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (N == 1) return std::bit_cast<T>(v);
                else if constexpr (N == 4) return std::bit_cast<uint16x4_t>(v);
                else return std::bit_cast<uint16x8_t>(v);
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (N == 1) return std::bit_cast<T>(v);
                else if constexpr (N == 2) return std::bit_cast<uint32x2_t>(v);
                else return std::bit_cast<uint32x4_t>(v);
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (N == 1) return std::bit_cast<uint64x1_t>(v);
                else return std::bit_cast<uint64x2_t>(v);
            }
        }
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(uint8x8_t const& v) noexcept -> Vec<8, std::uint8_t> {
        return std::bit_cast<Vec<8, std::uint8_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(uint8x16_t const& v) noexcept -> Vec<16, std::uint8_t> {
        return std::bit_cast<Vec<16, std::uint8_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int8x8_t const& v) noexcept -> Vec<8, std::int8_t> {
        return std::bit_cast<Vec<8, std::int8_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int8x16_t const& v) noexcept -> Vec<16, std::int8_t> {
        return std::bit_cast<Vec<16, std::int8_t>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(uint16x4_t const& v) noexcept -> Vec<4, std::uint16_t> {
        return std::bit_cast<Vec<4, std::uint16_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(uint16x8_t const& v) noexcept -> Vec<8, std::uint16_t> {
        return std::bit_cast<Vec<8, std::uint16_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int16x4_t const& v) noexcept -> Vec<4, std::int16_t> {
        return std::bit_cast<Vec<4, std::int16_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int16x8_t const& v) noexcept -> Vec<8, std::int16_t> {
        return std::bit_cast<Vec<8, std::int16_t>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(uint32x2_t const& v) noexcept -> Vec<2, std::uint32_t> {
        return std::bit_cast<Vec<2, std::uint32_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(uint32x4_t const& v) noexcept -> Vec<4, std::uint32_t> {
        return std::bit_cast<Vec<4, std::uint32_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int32x2_t const& v) noexcept -> Vec<2, std::int32_t> {
        return std::bit_cast<Vec<2, std::int32_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int32x4_t const& v) noexcept -> Vec<4, std::int32_t> {
        return std::bit_cast<Vec<4, std::int32_t>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(uint64x1_t const& v) noexcept -> Vec<1, std::uint64_t> {
        return std::bit_cast<Vec<1, std::uint64_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(uint64x2_t const& v) noexcept -> Vec<2, std::uint64_t> {
        return std::bit_cast<Vec<2, std::uint64_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int64x1_t const& v) noexcept -> Vec<1, std::int64_t> {
        return std::bit_cast<Vec<1, std::int64_t>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(int64x2_t const& v) noexcept -> Vec<2, std::int64_t> {
        return std::bit_cast<Vec<2, std::int64_t>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(float16x4_t const& v) noexcept -> Vec<4, ui::float16> {
        return std::bit_cast<Vec<4, ui::float16>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float16x8_t const& v) noexcept -> Vec<8, ui::float16> {
        return std::bit_cast<Vec<8, ui::float16>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(float32x2_t const& v) noexcept -> Vec<2, float> {
        return std::bit_cast<Vec<2, float>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float32x4_t const& v) noexcept -> Vec<4, float> {
        return std::bit_cast<Vec<4, float>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(float64x1_t const& v) noexcept -> Vec<1, double> {
        return std::bit_cast<Vec<1, double>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float64x2_t const& v) noexcept -> Vec<2, double> {
        return std::bit_cast<Vec<2, double>>(v);
    }

    namespace internal {

        template <typename To, std::size_t M0, std::size_t N, typename From>
        UI_ALWAYS_INLINE auto cast_helper(
            Vec<N, From> const& v,
            auto&& fn0
        ) noexcept -> Vec<N, To> {
            using ret_t = Vec<N, To>;
            
            if constexpr (M0 != 1 && N == 1) {
                return {
                    .val = static_cast<To>(v.val)
                };
            } else {
                if constexpr (N == M0) {
                    return std::bit_cast<ret_t>(fn0(v));
                } else {
                    return join(
                        cast_helper<To, M0>(v.lo, fn0),
                        cast_helper<To, M0>(v.hi, fn0)
                    );
                }
            }
        }

        template <typename To, std::size_t M0, std::size_t M1, std::size_t N, typename From>
        UI_ALWAYS_INLINE auto cast_helper(
            Vec<N, From> const& v,
            auto&& fn0,
            auto&& fn1
        ) noexcept -> Vec<N, To> {
            using ret_t = Vec<N, To>;
            
            if constexpr (M0 != 1 && N == 1) {
                return {
                    .val = static_cast<To>(v.val)
                };
            } else {
                if constexpr (N == M0) {
                    return std::bit_cast<ret_t>(fn0(v));
                } else if constexpr (N == M1) {
                    return std::bit_cast<ret_t>(fn1(v));
                } else {
                    return join(
                        cast_helper<To, M0, M1>(v.lo, fn0, fn1),
                        cast_helper<To, M0, M1>(v.hi, fn0, fn1)
                    );
                }
            }
        }
            /*__asm__(".global CAST_FUNCTION_MARKER_START");*/
            /*__asm__("CAST_FUNCTION_MARKER_START:");*/
            /*__asm__(".global CAST_FUNCTION_MARKER_END");*/
            /*__asm__("CAST_FUNCTION_MARKER_END:");*/
        template <typename To, bool Saturating = false>
        struct CastImpl {
            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::int8_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    return v;
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    return cast_helper<To, 8>(
                        v,
                        [](auto const& v) { return vmovl_s8(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return cast_helper<To, 4>(
                        temp,
                        [](auto const& v) { return vmovl_s16(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return cast_helper<To, 2>(
                        temp,
                        [](auto const& v) { return vmovl_s32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    return std::bit_cast<Vec<N, To>>(v);
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return cast_helper<To, 2, 4>(
                        temp,
                        [](auto const& v) { return vcvt_f32_s32(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f32_s32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return cast_helper<To, 1, 2>(
                        temp,
                        [](auto const& v) { return vcvt_f64_s64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_s64(to_vec(v)); }
                    ); 
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return cast_helper<To, 4, 8>(
                        temp,
                        [](auto const& v) { return vcvt_f16_s16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_s16(to_vec(v)); }
                    ); 
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::uint8_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    return std::bit_cast<Vec<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    auto temp = CastImpl<std::uint32_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::uint64_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    return v; 
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    return cast_helper<To, 8>(
                        v,
                        [](auto const& v) { return vmovl_u8(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return cast_helper<To, 4>(
                        temp,
                        [](auto const& v) { return vmovl_u16(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::uint32_t>{}(v); 
                    return cast_helper<To, 4>(
                        temp,
                        [](auto const& v) { return vmovl_u32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::uint32_t, Saturating>{}(v);
                    return cast_helper<To, 2, 4>(
                        temp,
                        [](auto const& v) { return vcvt_f32_u32(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f32_u32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    auto temp = CastImpl<std::uint64_t, Saturating>{}(v);
                    return cast_helper<To, 1, 2>(
                        temp,
                        [](auto const& v) { return vcvt_f64_u64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_u64(to_vec(v)); }
                    ); 
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                    return cast_helper<To, 4, 8>(
                        temp,
                        [](auto const& v) { return vcvt_f16_u16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_u16(to_vec(v)); }
                    ); 
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::int16_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    return cast_helper<To, 8>(
                        v,
                        [](auto const& v) {
                            if constexpr (Saturating) {
                                return vqmovn_s16(to_vec(v));
                            } else {
                                return vmovn_s16(to_vec(v));
                            }
                        }
                    );
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    return v;
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    return cast_helper<To, 4>(
                        v,
                        [](auto const& v) { return vmovl_s16(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v); 
                    return cast_helper<To, 2>(
                        temp,
                        [](auto const& v) { return vmovl_s32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    return cast_helper<To, 8>(
                        v,
                        [](auto const& v) {
                            if constexpr (Saturating) {
                                return vqmovun_s16(to_vec(v));
                            } else {
                                return vmovn_u16(std::bit_cast<uint16x8_t>(to_vec(v)));
                            }
                        }
                    );
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    return std::bit_cast<Vec<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return cast_helper<To, 2, 4>(
                        temp,
                        [](auto const& v) { return vcvt_f32_s32(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f32_s32(to_vec(v)); }
                    );
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return cast_helper<To, 1, 2>(
                        temp,
                        [](auto const& v) { return vcvt_f64_s64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_s64(to_vec(v)); }
                    );
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    return cast_helper<To, 4, 8>(
                        v,
                        [](auto const& v) { return vcvt_f16_s16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_s16(to_vec(v)); }
                    ); 
                }

            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::uint16_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::uint8_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    return std::bit_cast<Vec<N, To>>(v);
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    auto temp = CastImpl<std::uint32_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::uint64_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    return cast_helper<To, 8>(
                        v,
                        [](auto const& v) { return vqmovn_u16(to_vec(v)); }
                    );
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    return v; 
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    return cast_helper<To, 4>(
                        v,
                        [](auto const& v) { return vmovl_u16(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::uint32_t>{}(v); 
                    return cast_helper<To, 2>(
                        temp,
                        [](auto const& v) { return vmovl_u32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, float>) {
                    auto temp = CastImpl<std::uint32_t, Saturating>{}(v);
                    return cast_helper<To, 2, 4>(
                        temp,
                        [](auto const& v) { return vcvt_f32_u32(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f32_u32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    auto temp = CastImpl<std::uint64_t, Saturating>{}(v);
                    return cast_helper<To, 1, 2>(
                        temp,
                        [](auto const& v) { return vcvt_f64_u64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_u64(to_vec(v)); }
                    ); 
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    return cast_helper<To, 4, 8>(
                        v,
                        [](auto const& v) { return vcvt_f16_u16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_u16(to_vec(v)); }
                    ); 
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::int32_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return CastImpl<std::int8_t, Saturating>{}(temp);
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    return cast_helper<To, 4>(
                        v,
                        [](auto const& v) {
                            if constexpr (Saturating) {
                                return vqmovn_s32(to_vec(v));
                            } else {
                                return vmovn_s32(to_vec(v));
                            }
                        }
                    );
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    return v; 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    return cast<To, 2>(
                        v,
                        [](auto const& v) { return vmovl_s32(to_vec(v)); }
                    );
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                    return CastImpl<std::uint8_t, Saturating>{}(temp);
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    return cast_helper<To, 4>(
                        v,
                        [](auto const& v) {
                            if constexpr (Saturating) {
                                return vqmovun_s32(to_vec(v));
                            } else {
                                return vmovn_u32(std::bit_cast<uint32x4_t>(to_vec(v)));
                            }
                        }
                    );
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    return std::bit_cast<Vec<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, float>) {
                    return cast_helper<To, 2, 4>(
                        v,
                        [](auto const& v) { return vcvt_f32_s32(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f32_s32(to_vec(v)); }
                    );
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return cast_helper<To, 1, 2>(
                        temp,
                        [](auto const& v) { return vcvt_f64_s64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_s64(to_vec(v)); }
                    );
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return cast_helper<To, 4, 8>(
                        temp,
                        [](auto const& v) { return vcvt_f16_s16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_s16(to_vec(v)); }
                    ); 
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::uint32_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::uint8_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    return std::bit_cast<Vec<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::uint64_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return CastImpl<std::uint8_t>{}(temp);
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    return cast_helper<To, 4>(
                        v,
                        [](auto const& v) { return vqmovn_u32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    return v;
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    return cast_helper<To, 4>(
                        v,
                        [](auto const& v) { return vmovl_u32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, float>) {
                    return cast_helper<To, 2, 4>(
                        v,
                        [](auto const& v) { return vcvt_f32_u32(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f32_u32(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    auto temp = CastImpl<std::uint64_t, Saturating>{}(v);
                    return cast_helper<To, 1, 2>(
                        temp,
                        [](auto const& v) { return vcvt_f64_u64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_u64(to_vec(v)); }
                    ); 
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                    return cast_helper<To, 4, 8>(
                        temp,
                        [](auto const& v) { return vcvt_f16_u16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_u16(to_vec(v)); }
                    ); 
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::int64_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return CastImpl<std::int8_t, Saturating>{}(temp);
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return CastImpl<std::int16_t, Saturating>{}(temp);
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    return cast_helper<To, 2>(
                        v,
                        [](auto const& v) {
                            if constexpr (Saturating) {
                                return vqmovn_s64(to_vec(v));
                            } else {
                                return vmovn_s64(to_vec(v));
                            }
                        }
                    );
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    return v;
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    auto temp = CastImpl<std::int8_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    return cast_helper<To, 2>(
                        v,
                        [](auto const& v) {
                            if constexpr (Saturating) {
                                return vqmovun_s64(to_vec(v));
                            } else {
                                return vmovn_u64(std::bit_cast<int64x2_t>(to_vec(v)));
                            }
                        }
                    );
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    return std::bit_cast<Vec<N, To>>(v);
                } else if constexpr (std::same_as<To, float>) {
                    return map([](auto v) { return static_cast<float>(v); }, v); 
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    return cast_helper<To, 1, 2>(
                        v,
                        [](auto const& v) { return vcvt_f64_s64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_s64(to_vec(v)); }
                    );
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return cast_helper<To, 4, 8>(
                        temp,
                        [](auto const& v) { return vcvt_f16_s16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_s16(to_vec(v)); }
                    ); 
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, std::uint64_t> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::uint8_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<Vec<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    return std::bit_cast<Vec<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::uint8_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return CastImpl<std::uint8_t>{}(temp);
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    auto temp = CastImpl<std::uint32_t>{}(v); 
                    return CastImpl<std::uint16_t>{}(temp);
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    return cast_helper<To, 2>(
                        v,
                        [](auto const& v) { return vqmovn_u64(to_vec(v)); }
                    ); 
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    return v;
                } else if constexpr (std::same_as<To, float>) {
                    return map([](auto v) { return static_cast<double>(v); }, v);
                } else if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    auto temp = CastImpl<std::uint64_t, Saturating>{}(v);
                    return cast_helper<To, 1, 2>(
                        temp,
                        [](auto const& v) { return vcvt_f64_u64(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f64_u64(to_vec(v)); }
                    ); 
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    auto temp = CastImpl<std::uint16_t, Saturating>{}(v);
                    return cast_helper<To, 4, 8>(
                        temp,
                        [](auto const& v) { return vcvt_f16_u16(to_vec(v)); },
                        [](auto const& v) { return vcvtq_f16_u16(to_vec(v)); }
                    ); 
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, float> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, double>) {
                    #ifdef UI_CPU_ARM64
                    return cast_helper<To, 2>(
                        v,
                        [](auto const& v) { return vcvt_f64_f32(to_vec(v)); }
                    );
                    #else
                    return map([](auto v) { return static_cast<double>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, float>) {
                    return v;
                } else if constexpr (std::integral<To>) {
                    if constexpr (std::is_signed_v<To>) {
                        auto temp = cast_helper<std::int32_t, 2, 4>(
                            v,
                            [](auto const& v) { return vcvt_s32_f32(to_vec(v)); },
                            [](auto const& v) { return vcvtq_s32_f32(to_vec(v)); }
                        );
                        return CastImpl<To, Saturating>{}(temp);
                    } else {
                        auto temp = cast_helper<std::uint32_t, 2, 4>(
                            v,
                            [](auto const& v) { return vcvt_u32_f32(to_vec(v)); },
                            [](auto const& v) { return vcvtq_u32_f32(to_vec(v)); }
                        );
                        return CastImpl<To, Saturating>{}(temp);
                    }
                } else if constexpr (std::same_as<To, ui::float16>) {
                    return map([](auto v) { return ui::float16(v); }, v);
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                Vec<N, double> const& v
            ) noexcept -> Vec<N, To> {
                if constexpr (std::same_as<To, double>) {
                    return v;
                } else if constexpr (std::same_as<To, float>) {
                    #ifdef UI_CPU_ARM64
                    return cast_helper<To, 2>(
                        v,
                        [](auto const& v) { return vcvt_f32_f64(to_vec(v)); }
                    );
                    #else
                    return map([](auto v) { return static_cast<float>(v); }, v); 
                    #endif
                } else if constexpr (std::integral<To>) {
                    #ifdef UI_CPU_ARM64
                    if constexpr (std::is_signed_v<To>) {
                        auto temp = cast_helper<std::int64_t, 2>(
                            v,
                            [](auto const& v) { return vcvtq_s64_f64(to_vec(v)); }
                        );
                        return CastImpl<To, Saturating>{}(temp);
                    } else {
                        auto temp = cast_helper<std::uint64_t, 2>(
                            v,
                            [](auto const& v) { return vcvtq_u64_f64(to_vec(v)); }
                        );
                        return CastImpl<To, Saturating>{}(temp);
                    }
                    #else
                    return map([](auto v) { return static_cast<To>(v); }, v); 
                    #endif
                } else if constexpr (std::same_as<To, ui::float16>) {
                    return map([](auto v) { return ui::float16(v); }, v);
                }
            }
        };
    } // namespace internal

    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return internal::CastImpl<To, false>{}(v);
    }

    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto sat_cast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return internal::CastImpl<To, true>{}(v);
    }

    // retinterpret cast
    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE constexpr auto rcast(Vec<N, From> const& v) noexcept -> Vec<N, To> {
        return std::bit_cast<Vec<N, To>>(v);
    }

    template <typename T>
    UI_ALWAYS_INLINE constexpr auto from_vec(auto const& v) noexcept {
        return rcast<T>(from_vec(v));
    }


    template <std::size_t R, std::size_t C, typename T>
    UI_ALWAYS_INLINE constexpr auto to_vec(VecMat<R, C, T> const& m) noexcept {
        if constexpr (std::same_as<T, float16>) {
            if constexpr (C == 4) {
                if constexpr (R == 2) {
                    return std::bit_cast<float16x4x2_t>(m);
                } else if constexpr (R == 3) {
                    return std::bit_cast<float16x4x3_t>(m);
                } else if constexpr (R == 4) {
                    return std::bit_cast<float16x4x4_t>(m);
                }
            } else if constexpr (C == 8) {
                if constexpr (R == 2) {
                    return std::bit_cast<float16x8x2_t>(m);
                } else if constexpr (R == 3) {
                    return std::bit_cast<float16x8x3_t>(m);
                } else if constexpr (R == 4) {
                    return std::bit_cast<float16x8x4_t>(m);
                }
            }
        } else if constexpr (std::same_as<T, float>) {
            if constexpr (C == 2) {
                if constexpr (R == 2) {
                    return std::bit_cast<float32x2x2_t>(m);
                } else if constexpr (R == 3) {
                    return std::bit_cast<float32x2x3_t>(m);
                } else if constexpr (R == 4) {
                    return std::bit_cast<float32x2x4_t>(m);
                }
            } else if constexpr (C == 4) {
                if constexpr (R == 2) {
                    return std::bit_cast<float32x4x2_t>(m);
                } else if constexpr (R == 3) {
                    return std::bit_cast<float32x4x3_t>(m);
                } else if constexpr (R == 4) {
                    return std::bit_cast<float32x4x4_t>(m);
                }
            }
        } else if constexpr (std::same_as<T, double>) {
            if constexpr (C == 1) {
                if constexpr (R == 2) {
                    return std::bit_cast<float64x1x2_t>(m);
                } else if constexpr (R == 3) {
                    return std::bit_cast<float64x1x3_t>(m);
                } else if constexpr (R == 4) {
                    return std::bit_cast<float64x1x4_t>(m);
                }
            } else if constexpr (C == 2) {
                if constexpr (R == 2) {
                    return std::bit_cast<float64x2x2_t>(m);
                } else if constexpr (R == 3) {
                    return std::bit_cast<float64x2x3_t>(m);
                } else if constexpr (R == 4) {
                    return std::bit_cast<float64x2x4_t>(m);
                }
            }
        } else if constexpr (std::is_signed_v<T>) {
            if constexpr (sizeof(T) == 1) {
                if constexpr (C == 8) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int8x8x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int8x8x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int8x8x4_t>(m);
                    }
                } else if constexpr (C == 16) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int8x16x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int8x16x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int8x16x4_t>(m);
                    }
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (C == 4) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int16x4x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int16x4x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int16x4x4_t>(m);
                    }
                } else if constexpr (C == 8) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int16x8x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int16x8x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int16x8x4_t>(m);
                    }
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (C == 2) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int32x2x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int32x2x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int32x2x4_t>(m);
                    }
                } else if constexpr (C == 4) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int32x4x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int32x4x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int32x4x4_t>(m);
                    }
                }
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (C == 1) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int64x1x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int64x1x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int64x1x4_t>(m);
                    }
                } else if constexpr (C == 2) {
                    if constexpr (R == 2) {
                        return std::bit_cast<int64x2x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<int64x2x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<int64x2x4_t>(m);
                    }
                }
            }
        } else {
            if constexpr (sizeof(T) == 1) {
                if constexpr (C == 8) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint8x8x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint8x8x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint8x8x4_t>(m);
                    }
                } else if constexpr (C == 16) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint8x16x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint8x16x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint8x16x4_t>(m);
                    }
                }
            } else if constexpr (sizeof(T) == 2) {
                if constexpr (C == 4) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint16x4x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint16x4x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint16x4x4_t>(m);
                    }
                } else if constexpr (C == 8) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint16x8x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint16x8x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint16x8x4_t>(m);
                    }
                }
            } else if constexpr (sizeof(T) == 4) {
                if constexpr (C == 2) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint32x2x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint32x2x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint32x2x4_t>(m);
                    }
                } else if constexpr (C == 4) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint32x4x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint32x4x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint32x4x4_t>(m);
                    }
                }
            } else if constexpr (sizeof(T) == 8) {
                if constexpr (C == 1) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint64x1x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint64x1x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint64x1x4_t>(m);
                    }
                } else if constexpr (C == 2) {
                    if constexpr (R == 2) {
                        return std::bit_cast<uint64x2x2_t>(m);
                    } else if constexpr (R == 3) {
                        return std::bit_cast<uint64x2x3_t>(m);
                    } else if constexpr (R == 4) {
                        return std::bit_cast<uint64x2x4_t>(m);
                    }
                }
            }
        }
    }

    template <std::integral T>
        requires (sizeof(T) == 1)
    UI_ALWAYS_INLINE constexpr auto from_vec(T const& v) noexcept {
        if constexpr (std::is_signed_v<T>) {
            if constexpr        (std::same_as<T, int8x8x2_t>) {
                return std::bit_cast<VecMat<2,  8, std::int8_t>>(v);
            } else if constexpr (std::same_as<T, int8x8x3_t>) {
                return std::bit_cast<VecMat<3,  8, std::int8_t>>(v);
            } else if constexpr (std::same_as<T, int8x8x4_t>) {
                return std::bit_cast<VecMat<4,  8, std::int8_t>>(v);
            } else if constexpr (std::same_as<T, int8x16x2_t>) {
                return std::bit_cast<VecMat<2, 16, std::int8_t>>(v);
            } else if constexpr (std::same_as<T, int8x16x3_t>) {
                return std::bit_cast<VecMat<3, 16, std::int8_t>>(v);
            } else if constexpr (std::same_as<T, int8x16x4_t>) {
                return std::bit_cast<VecMat<4, 16, std::int8_t>>(v);
            }
        } else {
            if constexpr        (std::same_as<T, uint8x8x2_t>) {
                return std::bit_cast<VecMat<2,  8, std::uint8_t>>(v);
            } else if constexpr (std::same_as<T, uint8x8x3_t>) {
                return std::bit_cast<VecMat<3,  8, std::uint8_t>>(v);
            } else if constexpr (std::same_as<T, uint8x8x4_t>) {
                return std::bit_cast<VecMat<4,  8, std::uint8_t>>(v);
            } else if constexpr (std::same_as<T, uint8x16x2_t>) {
                return std::bit_cast<VecMat<2, 16, std::uint8_t>>(v);
            } else if constexpr (std::same_as<T, uint8x16x3_t>) {
                return std::bit_cast<VecMat<3, 16, std::uint8_t>>(v);
            } else if constexpr (std::same_as<T, uint8x16x4_t>) {
                return std::bit_cast<VecMat<4, 16, std::uint8_t>>(v);
            }
        }
    }

    template <std::integral T>
        requires (sizeof(T) == 2)
    UI_ALWAYS_INLINE constexpr auto from_vec(T const& v) noexcept {
        if constexpr (std::is_signed_v<T>) {
            if constexpr        (std::same_as<T, int16x4x2_t>) {
                return std::bit_cast<VecMat<2, 4, std::int16_t>>(v);
            } else if constexpr (std::same_as<T, int16x4x3_t>) {
                return std::bit_cast<VecMat<3, 4, std::int16_t>>(v);
            } else if constexpr (std::same_as<T, int16x4x4_t>) {
                return std::bit_cast<VecMat<4, 4, std::int16_t>>(v);
            } else if constexpr (std::same_as<T, int16x8x2_t>) {
                return std::bit_cast<VecMat<2, 8, std::int16_t>>(v);
            } else if constexpr (std::same_as<T, int16x8x3_t>) {
                return std::bit_cast<VecMat<3, 8, std::int16_t>>(v);
            } else if constexpr (std::same_as<T, int16x8x4_t>) {
                return std::bit_cast<VecMat<4, 8, std::int16_t>>(v);
            }
        } else {
            if constexpr        (std::same_as<T, uint16x4x2_t>) {
                return std::bit_cast<VecMat<2, 4, std::uint16_t>>(v);
            } else if constexpr (std::same_as<T, uint16x4x3_t>) {
                return std::bit_cast<VecMat<3, 4, std::uint16_t>>(v);
            } else if constexpr (std::same_as<T, uint16x4x4_t>) {
                return std::bit_cast<VecMat<4, 4, std::uint16_t>>(v);
            } else if constexpr (std::same_as<T, uint16x8x2_t>) {
                return std::bit_cast<VecMat<2, 8, std::uint16_t>>(v);
            } else if constexpr (std::same_as<T, uint16x8x3_t>) {
                return std::bit_cast<VecMat<3, 8, std::uint16_t>>(v);
            } else if constexpr (std::same_as<T, uint16x8x4_t>) {
                return std::bit_cast<VecMat<4, 8, std::uint16_t>>(v);
            }
        }
    }

    template <std::integral T>
        requires (sizeof(T) == 4)
    UI_ALWAYS_INLINE constexpr auto from_vec(T const& v) noexcept {
        if constexpr (std::is_signed_v<T>) {
            if constexpr        (std::same_as<T, int32x2x2_t>) {
                return std::bit_cast<VecMat<2, 2, std::int32_t>>(v);
            } else if constexpr (std::same_as<T, int32x2x3_t>) {
                return std::bit_cast<VecMat<3, 2, std::int32_t>>(v);
            } else if constexpr (std::same_as<T, int32x2x4_t>) {
                return std::bit_cast<VecMat<4, 2, std::int32_t>>(v);
            } else if constexpr (std::same_as<T, int32x4x2_t>) {
                return std::bit_cast<VecMat<2, 4, std::int32_t>>(v);
            } else if constexpr (std::same_as<T, int32x4x3_t>) {
                return std::bit_cast<VecMat<3, 4, std::int32_t>>(v);
            } else if constexpr (std::same_as<T, int32x4x4_t>) {
                return std::bit_cast<VecMat<4, 4, std::int32_t>>(v);
            }
        } else {
            if constexpr        (std::same_as<T, uint32x2x2_t>) {
                return std::bit_cast<VecMat<2, 2, std::uint32_t>>(v);
            } else if constexpr (std::same_as<T, uint32x2x3_t>) {
                return std::bit_cast<VecMat<3, 2, std::uint32_t>>(v);
            } else if constexpr (std::same_as<T, uint32x2x4_t>) {
                return std::bit_cast<VecMat<4, 2, std::uint32_t>>(v);
            } else if constexpr (std::same_as<T, uint32x4x2_t>) {
                return std::bit_cast<VecMat<2, 4, std::uint32_t>>(v);
            } else if constexpr (std::same_as<T, uint32x4x3_t>) {
                return std::bit_cast<VecMat<3, 4, std::uint32_t>>(v);
            } else if constexpr (std::same_as<T, uint32x4x4_t>) {
                return std::bit_cast<VecMat<4, 4, std::uint32_t>>(v);
            }
        }
    }

    template <std::integral T>
        requires (sizeof(T) == 8)
    UI_ALWAYS_INLINE constexpr auto from_vec(T const& v) noexcept {
        if constexpr (std::is_signed_v<T>) {
            if constexpr        (std::same_as<T, int64x1x2_t>) {
                return std::bit_cast<VecMat<2, 1, std::int64_t>>(v);
            } else if constexpr (std::same_as<T, int64x1x3_t>) {
                return std::bit_cast<VecMat<3, 1, std::int64_t>>(v);
            } else if constexpr (std::same_as<T, int64x1x4_t>) {
                return std::bit_cast<VecMat<4, 1, std::int64_t>>(v);
            } else if constexpr (std::same_as<T, int64x2x2_t>) {
                return std::bit_cast<VecMat<2, 2, std::int64_t>>(v);
            } else if constexpr (std::same_as<T, int64x2x3_t>) {
                return std::bit_cast<VecMat<3, 2, std::int64_t>>(v);
            } else if constexpr (std::same_as<T, int64x2x4_t>) {
                return std::bit_cast<VecMat<4, 2, std::int64_t>>(v);
            }
        } else {
            if constexpr        (std::same_as<T, uint64x1x2_t>) {
                return std::bit_cast<VecMat<2, 1, std::uint64_t>>(v);
            } else if constexpr (std::same_as<T, uint64x1x3_t>) {
                return std::bit_cast<VecMat<3, 1, std::uint64_t>>(v);
            } else if constexpr (std::same_as<T, uint64x1x4_t>) {
                return std::bit_cast<VecMat<4, 1, std::uint64_t>>(v);
            } else if constexpr (std::same_as<T, uint64x2x2_t>) {
                return std::bit_cast<VecMat<2, 2, std::uint64_t>>(v);
            } else if constexpr (std::same_as<T, uint64x2x3_t>) {
                return std::bit_cast<VecMat<3, 2, std::uint64_t>>(v);
            } else if constexpr (std::same_as<T, uint64x2x4_t>) {
                return std::bit_cast<VecMat<4, 2, std::uint64_t>>(v);
            }
        }
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(float16x4x2_t const& v) noexcept {
        return std::bit_cast<VecMat<2, 4, float16>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float16x4x3_t const& v) noexcept {
        return std::bit_cast<VecMat<3, 4, float16>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float16x4x4_t const& v) noexcept {
        return std::bit_cast<VecMat<4, 4, float16>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float16x8x2_t const& v) noexcept {
        return std::bit_cast<VecMat<2, 8, float16>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float16x8x3_t const& v) noexcept {
        return std::bit_cast<VecMat<3, 8, float16>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float16x8x4_t const& v) noexcept {
        return std::bit_cast<VecMat<4, 8, float16>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(float32x2x2_t const& v) noexcept {
        return std::bit_cast<VecMat<2, 2, float>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float32x2x3_t const& v) noexcept {
        return std::bit_cast<VecMat<3, 2, float>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float32x2x4_t const& v) noexcept {
        return std::bit_cast<VecMat<4, 2, float>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float32x4x2_t const& v) noexcept {
        return std::bit_cast<VecMat<2, 4, float>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float32x4x3_t const& v) noexcept {
        return std::bit_cast<VecMat<3, 4, float>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float32x4x4_t const& v) noexcept {
        return std::bit_cast<VecMat<4, 4, float>>(v);
    }

    UI_ALWAYS_INLINE constexpr auto from_vec(float64x1x2_t const& v) noexcept {
        return std::bit_cast<VecMat<2, 1, double>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float64x1x3_t const& v) noexcept {
        return std::bit_cast<VecMat<3, 1, double>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float64x1x4_t const& v) noexcept {
        return std::bit_cast<VecMat<4, 1, double>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float64x2x2_t const& v) noexcept {
        return std::bit_cast<VecMat<2, 2, double>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float64x2x3_t const& v) noexcept {
        return std::bit_cast<VecMat<3, 2, double>>(v);
    }
    UI_ALWAYS_INLINE constexpr auto from_vec(float64x2x4_t const& v) noexcept {
        return std::bit_cast<VecMat<4, 2, double>>(v);
    }

} // namespace ui::arm::neon

#endif // AMT_UI_ARCH_ARM_CAST_HPP

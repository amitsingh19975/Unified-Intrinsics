#ifndef AMT_UI_ARCH_ARM_CAST_HPP
#define AMT_UI_ARCH_ARM_CAST_HPP

#include "../../base_vec.hpp"
#include "../../base.hpp"
#include "../../vec_headers.hpp"
#include <bit>
#include <concepts>
#include <cstdint>

namespace ui::arm {
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto to_vec(VecReg<N, T> const& v) noexcept {
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<float, T>) {
                if constexpr (N == 1) return std::bit_cast<float>(v);
                else if constexpr (N == 2) return std::bit_cast<float32x2_t>(v);
                else return std::bit_cast<float32x4_t>(v);
            } else if constexpr (std::same_as<double, T>) {
                if constexpr (N == 1) return std::bit_cast<float64x1_t>(v);
                else return std::bit_cast<float64x2_t>(v);
            } else {
                static_assert(sizeof(T) == sizeof(float) || sizeof(T) == sizeof(double), "Unknow floating-point type, expecting 'float' or 'double'");
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

    namespace internal {

        template <typename To, std::size_t M0, std::size_t N, typename From>
        UI_ALWAYS_INLINE auto cast_helper(
            VecReg<N, From> const& v,
            auto&& fn0
        ) noexcept -> VecReg<N, To> {
            using ret_t = VecReg<N, To>;
            
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
            VecReg<N, From> const& v,
            auto&& fn0,
            auto&& fn1
        ) noexcept -> VecReg<N, To> {
            using ret_t = VecReg<N, To>;
            
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
                VecReg<N, std::int8_t> const& v
            ) noexcept -> VecReg<N, To> {
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
                    return std::bit_cast<VecReg<N, To>>(v);
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return std::bit_cast<VecReg<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return std::bit_cast<VecReg<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return std::bit_cast<VecReg<N, To>>(temp);
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, std::uint8_t> const& v
            ) noexcept -> VecReg<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    return std::bit_cast<VecReg<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    auto temp = CastImpl<std::uint32_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::uint64_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, std::int16_t> const& v
            ) noexcept -> VecReg<N, To> {
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
                    return std::bit_cast<VecReg<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::uint32_t>) {
                    auto temp = CastImpl<std::int32_t, Saturating>{}(v);
                    return std::bit_cast<VecReg<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return std::bit_cast<VecReg<N, To>>(temp); 
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
                }

            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, std::uint16_t> const& v
            ) noexcept -> VecReg<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::uint8_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    return std::bit_cast<VecReg<N, To>>(v);
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    auto temp = CastImpl<std::uint32_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::uint64_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, std::int32_t> const& v
            ) noexcept -> VecReg<N, To> {
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
                    return std::bit_cast<VecReg<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::uint64_t>) {
                    auto temp = CastImpl<std::int64_t, Saturating>{}(v);
                    return std::bit_cast<VecReg<N, To>>(temp);
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, std::uint32_t> const& v
            ) noexcept -> VecReg<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::uint8_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    return std::bit_cast<VecReg<N, To>>(v); 
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    auto temp = CastImpl<std::uint64_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, std::int64_t> const& v
            ) noexcept -> VecReg<N, To> {
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
                    return std::bit_cast<VecReg<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::uint16_t>) {
                    auto temp = CastImpl<std::int16_t, Saturating>{}(v);
                    return std::bit_cast<VecReg<N, To>>(temp);
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
                    return std::bit_cast<VecReg<N, To>>(v);
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, std::uint64_t> const& v
            ) noexcept -> VecReg<N, To> {
                if constexpr (std::same_as<To, std::int8_t>) {
                    auto temp = CastImpl<std::uint8_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp); 
                } else if constexpr (std::same_as<To, std::int16_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::int32_t>) {
                    auto temp = CastImpl<std::uint16_t>{}(v); 
                    return std::bit_cast<VecReg<N, To>>(temp);
                } else if constexpr (std::same_as<To, std::int64_t>) {
                    return std::bit_cast<VecReg<N, To>>(v); 
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, float> const& v
            ) noexcept -> VecReg<N, To> {
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
                }
            }

            template <std::size_t N>
            UI_ALWAYS_INLINE auto operator()(
                VecReg<N, double> const& v
            ) noexcept -> VecReg<N, To> {
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
                }
            }
        };
    } // namespace internal

    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto cast(VecReg<N, From> const& v) noexcept -> VecReg<N, To> {
        return internal::CastImpl<To, false>{}(v);
    }

    template <typename To, std::size_t N, typename From>
    UI_ALWAYS_INLINE auto sat_cast(VecReg<N, From> const& v) noexcept -> VecReg<N, To> {
        return internal::CastImpl<To, true>{}(v);
    }
} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_CAST_HPP

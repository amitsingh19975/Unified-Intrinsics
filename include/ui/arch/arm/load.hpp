#ifndef AMT_UI_ARCH_ARM_LOAD_HPP
#define AMT_UI_ARCH_ARM_LOAD_HPP

#include "cast.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ui::arm::neon { 
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto load(T val) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(val) };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec(vdup_n_f32(val));
                } else if constexpr (N == 4) {
                    return from_vec(vdupq_n_f32(val));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec(vdupq_n_f64(val));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vdup_n_f16(std::bit_cast<float16_t>(val)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vdupq_n_f16(std::bit_cast<float16_t>(val)));
                }
                #else
                return std::bit_cast<Vec<N, T>>(load<N>(std::bit_cast<std::uint16_t>(val.data)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                #ifdef UI_HAS_BFLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vdup_n_bf16(std::bit_cast<bfloat16_t>(val)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vdupq_n_bf16(std::bit_cast<bfloat16_t>(val)));
                }
                #else
                return std::bit_cast<Vec<N, T>>(load<N>(std::bit_cast<std::uint16_t>(val.data)));
                #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(vdup_n_s8(val));
                    } else if constexpr (N == 16) {
                        return from_vec(vdupq_n_s8(val));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(vdup_n_s16(val));
                    } else if constexpr (N == 8) {
                        return from_vec(vdupq_n_s16(val));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_n_s32(val));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_n_s32(val));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec(vdupq_n_s64(val));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(vdup_n_u8(val));
                    } else if constexpr (N == 16) {
                        return from_vec(vdupq_n_u8(val));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(vdup_n_u16(val));
                    } else if constexpr (N == 8) {
                        return from_vec(vdupq_n_u16(val));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_n_u32(val));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_n_u32(val));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec(vdupq_n_u64(val));
                    }
                }
            }
            auto t = load<N / 2>(val);
            return join(t, t);
        }
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto load(
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(v[Lane]) };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_lane_f32(to_vec(v), Lane));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_lane_f32(to_vec(v), Lane));
                    }
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_laneq_f32(to_vec(v), Lane));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_laneq_f32(to_vec(v), Lane));
                    }
                } else if constexpr (M > 4) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                } 
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (M == 1) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_lane_f64(to_vec(v), Lane));
                    }
                } else if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec(vdupq_laneq_f64(to_vec(v), Lane));
                    }
                } else if constexpr (M > 2) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                } 
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (M == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_lane_f16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_lane_f16(to_vec(v), Lane));
                    }
                } else if constexpr (M == 8) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_laneq_f16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_laneq_f16(to_vec(v), Lane));
                    }
                } else if constexpr (M > 8) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                }
                #else
                auto temp = rcast<std::uint16_t>(v);
                return rcast<T>(load<N, Lane>(temp));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                #ifdef UI_HAS_BFLOAT_16
                if constexpr (M == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_lane_bf16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_lane_bf16(to_vec(v), Lane));
                    }
                } else if constexpr (M == 8) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_laneq_bf16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_laneq_bf16(to_vec(v), Lane));
                    }
                } else if constexpr (M > 8) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                }
                #else
                auto temp = rcast<std::uint16_t>(v);
                return rcast<T>(load<N, Lane>(temp));
                #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (M == 8) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_lane_s8(to_vec(v), Lane));
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_lane_s8(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 16) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_laneq_s8(to_vec(v), Lane));
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_laneq_s8(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 16) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_lane_s16(to_vec(v), Lane));
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_lane_s16(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_laneq_s16(to_vec(v), Lane));
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_laneq_s16(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 8) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_lane_s32(to_vec(v), Lane));
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_lane_s32(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_laneq_s32(to_vec(v), Lane));
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_laneq_s32(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (M == 1) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_lane_s64(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_laneq_s64(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                #endif
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (M == 8) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_lane_u8(to_vec(v), Lane));
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_lane_u8(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 16) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_laneq_u8(to_vec(v), Lane));
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_laneq_u8(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 16) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_lane_u16(to_vec(v), Lane));
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_lane_u16(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_laneq_u16(to_vec(v), Lane));
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_laneq_u16(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 8) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_lane_u32(to_vec(v), Lane));
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_lane_u32(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_laneq_u32(to_vec(v), Lane));
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_laneq_u32(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (M == 1) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_lane_u64(to_vec(v), Lane));
                        }
                    } else if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_laneq_u64(to_vec(v), Lane));
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                #endif
                }
            }
            auto t = load<N / 2, Lane>(v);
            return join(t, t);
        }
    }


    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b
    ) noexcept {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                auto res = from_vec(vld2_f64(data));
                a = res.val[0];
                b = res.val[1];
                return;
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 8) {
                    auto res = from_vec(vld2_s64(data));
                    a = res.val[0];
                    b = res.val[1];
                    return;
                }
            } else {
                if constexpr (sizeof(T) == 8) {
                    auto res = from_vec(vld2_u64(data));
                    a = res.val[0];
                    b = res.val[1];
                    return;
                }
            }
            #endif
            a.val = data[0];
            b.val = data[1];
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    auto res = from_vec(vld2_f32(data));
                    a = res.val[0];
                    b = res.val[1];
                    return;
                } else if constexpr (N == 4) {
                    auto res = from_vec(vld2q_f32(data));
                    a = res.val[0];
                    b = res.val[1];
                    return;
                } 
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    auto res = from_vec(vld2q_f64(data));
                    a = res.val[0];
                    b = res.val[1];
                    return;
                } 
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    auto res = from_vec(vld2_f16(reinterpret_cast<float16_t const*>(data)));
                    a = res.val[0];
                    b = res.val[1];
                    return;
                } else if constexpr (N == 8) {
                    auto res = from_vec(vld2q_f16(reinterpret_cast<float16_t const*>(data)));
                    a = res.val[0];
                    b = res.val[1];
                    return;
                } 
                #else
                auto a0 = Vec<N, std::uint16_t>{};
                auto b0 = Vec<N, std::uint16_t>{};
                strided_load(reinterpret_cast<std::uint16_t const*>(data), a0, b0);
                a = rcast<float16>(a0);
                b = rcast<float16>(b0);
                return;
                #endif

            } else if constexpr (std::same_as<T, bfloat16>) {
                auto a0 = Vec<N, std::uint16_t>{};
                auto b0 = Vec<N, std::uint16_t>{};
                strided_load(reinterpret_cast<std::uint16_t const*>(data), a0, b0);
                a = rcast<bfloat16>(a0);
                b = rcast<bfloat16>(b0);
                return;
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld2_s8(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    } else if constexpr (N == 16) {
                        auto res = from_vec(vld2q_s8(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto res = from_vec(vld2_s16(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    } else if constexpr (N == 8) {
                        auto res = from_vec(vld2q_s16(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld2_s32(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    } else if constexpr (N == 4) {
                        auto res = from_vec(vld2q_s32(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld2q_s64(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                #endif
                }
            } else {
                 if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld2_u8(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    } else if constexpr (N == 16) {
                        auto res = from_vec(vld2q_u8(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto res = from_vec(vld2_u16(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    } else if constexpr (N == 8) {
                        auto res = from_vec(vld2q_u16(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld2_u32(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    } else if constexpr (N == 4) {
                        auto res = from_vec(vld2q_u32(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld2q_u64(data));
                        a = res.val[0];
                        b = res.val[1];
                        return;
                    }
                #endif
                }

            }
            strided_load(data, a.lo, b.lo);
            strided_load(data + N / 2 * 2, a.hi, b.hi);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c
    ) noexcept {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                auto res = from_vec(vld3_f64(data));
                a = res.val[0];
                b = res.val[1];
                c = res.val[2];
                return;
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 8) {
                    auto res = from_vec(vld3_s64(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    return;
                }
            } else {
                if constexpr (sizeof(T) == 8) {
                    auto res = from_vec(vld3_u64(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    return;
                }
            }
            #endif
            a.val = data[0];
            b.val = data[1];
            c.val = data[2];
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    auto res = from_vec(vld3_f32(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    return;
                } else if constexpr (N == 4) {
                    auto res = from_vec(vld3q_f32(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    return;
                } 
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    auto res = from_vec(vld3q_f64(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    return;
                } 
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    auto res = from_vec(vld3_f16(reinterpret_cast<float16_t const*>(data)));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    return;
                } else if constexpr (N == 8) {
                    auto res = from_vec(vld3q_f16(reinterpret_cast<float16_t const*>(data)));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    return;
                } 
                #else
                auto a0 = Vec<N, std::uint16_t>{};
                auto b0 = Vec<N, std::uint16_t>{};
                auto c0 = Vec<N, std::uint16_t>{};
                strided_load(reinterpret_cast<std::uint16_t const*>(data), a0, b0, c0);
                a = rcast<float16>(a0);
                b = rcast<float16>(b0);
                c = rcast<float16>(c0);
                return;
                #endif

            } else if constexpr (std::same_as<T, bfloat16>) {
                auto a0 = Vec<N, std::uint16_t>{};
                auto b0 = Vec<N, std::uint16_t>{};
                auto c0 = Vec<N, std::uint16_t>{};
                strided_load(reinterpret_cast<std::uint16_t const*>(data), a0, b0, c0);
                a = rcast<bfloat16>(a0);
                b = rcast<bfloat16>(b0);
                c = rcast<bfloat16>(c0);
                return;
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld3_s8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    } else if constexpr (N == 16) {
                        auto res = from_vec(vld3q_s8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto res = from_vec(vld3_s16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    } else if constexpr (N == 8) {
                        auto res = from_vec(vld3q_s16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld3_s32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    } else if constexpr (N == 4) {
                        auto res = from_vec(vld3q_s32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld3q_s64(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                #endif
                }
            } else {
                 if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld3_u8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    } else if constexpr (N == 16) {
                        auto res = from_vec(vld3q_u8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto res = from_vec(vld3_u16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    } else if constexpr (N == 8) {
                        auto res = from_vec(vld3q_u16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld3_u32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    } else if constexpr (N == 4) {
                        auto res = from_vec(vld3q_u32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld3q_u64(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        return;
                    }
                #endif
                }

            }
            strided_load(data, a.lo, b.lo, c.lo);
            strided_load(data + N / 2 * 3, a.hi, b.hi, c.hi);
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto strided_load(
        T const* UI_RESTRICT data,
        Vec<N, T>& UI_RESTRICT a,
        Vec<N, T>& UI_RESTRICT b,
        Vec<N, T>& UI_RESTRICT c,
        Vec<N, T>& UI_RESTRICT d
    ) noexcept {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                auto res = from_vec(vld4_f64(data));
                a = res.val[0];
                b = res.val[1];
                c = res.val[2];
                d = res.val[3];
                return;
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 8) {
                    auto res = from_vec(vld4_s64(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    d = res.val[3];
                    return;
                }
            } else {
                if constexpr (sizeof(T) == 8) {
                    auto res = from_vec(vld4_u64(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    d = res.val[3];
                    return;
                }
            }
            #endif
            a.val = data[0];
            b.val = data[1];
            c.val = data[2];
            d.val = data[3];
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    auto res = from_vec(vld4_f32(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    d = res.val[3];
                    return;
                } else if constexpr (N == 4) {
                    auto res = from_vec(vld4q_f32(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    d = res.val[3];
                    return;
                } 
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    auto res = from_vec(vld4q_f64(data));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    d = res.val[3];
                    return;
                } 
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    auto res = from_vec(vld4_f16(reinterpret_cast<float16_t const*>(data)));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    d = res.val[3];
                    return;
                } else if constexpr (N == 8) {
                    auto res = from_vec(vld4q_f16(reinterpret_cast<float16_t const*>(data)));
                    a = res.val[0];
                    b = res.val[1];
                    c = res.val[2];
                    d = res.val[3];
                    return;
                } 
                #else
                auto a0 = Vec<N, std::uint16_t>{};
                auto b0 = Vec<N, std::uint16_t>{};
                auto c0 = Vec<N, std::uint16_t>{};
                auto d0 = Vec<N, std::uint16_t>{};
                strided_load(reinterpret_cast<std::uint16_t const*>(data), a0, b0, c0, d0);
                a = rcast<float16>(a0);
                b = rcast<float16>(b0);
                c = rcast<float16>(c0);
                d = rcast<float16>(d0);
                return;
                #endif

            } else if constexpr (std::same_as<T, bfloat16>) {
                auto a0 = Vec<N, std::uint16_t>{};
                auto b0 = Vec<N, std::uint16_t>{};
                auto c0 = Vec<N, std::uint16_t>{};
                auto d0 = Vec<N, std::uint16_t>{};
                strided_load(reinterpret_cast<std::uint16_t const*>(data), a0, b0, c0, d0);
                a = rcast<bfloat16>(a0);
                b = rcast<bfloat16>(b0);
                c = rcast<bfloat16>(c0);
                d = rcast<bfloat16>(d0);
                return;
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld4_s8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    } else if constexpr (N == 16) {
                        auto res = from_vec(vld4q_s8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto res = from_vec(vld4_s16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    } else if constexpr (N == 8) {
                        auto res = from_vec(vld4q_s16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld4_s32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    } else if constexpr (N == 4) {
                        auto res = from_vec(vld4q_s32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld4q_s64(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                #endif
                }
            } else {
                 if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        auto res = from_vec(vld4_u8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    } else if constexpr (N == 16) {
                        auto res = from_vec(vld4q_u8(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        auto res = from_vec(vld4_u16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    } else if constexpr (N == 8) {
                        auto res = from_vec(vld4q_u16(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld4_u32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    } else if constexpr (N == 4) {
                        auto res = from_vec(vld4q_u32(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        auto res = from_vec(vld4q_u64(data));
                        a = res.val[0];
                        b = res.val[1];
                        c = res.val[2];
                        d = res.val[3];
                        return;
                    }
                #endif
                }

            }
            strided_load(data, a.lo, b.lo, c.lo, d.lo);
            strided_load(data + N / 2 * 4, a.hi, b.hi, c.hi, d.hi);
        }
    }

} // namespace ui::arm::neon;

#endif // AMT_UI_ARCH_ARM_LOAD_HPP

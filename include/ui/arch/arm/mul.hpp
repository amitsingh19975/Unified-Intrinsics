#ifndef AMT_UI_ARCH_ARM_MUL_HPP
#define AMT_UI_ARCH_ARM_MUL_HPP

#include "cast.hpp"
#include "basic.hpp"
#include "ui/base.hpp"
#include "ui/float.hpp"
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace ui::arm::neon {
// MARK: Multiplication
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) return { .val = static_cast<T>(lhs.val * rhs.val) };
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return from_vec<T>(vmul_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4) 
                    return from_vec<T>(vmulq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                    if constexpr (N == 2)
                        return from_vec<T>(vmulq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vmul_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vmulq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(mul(cast<float>(lhs), cast<float>(rhs)));    
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(mul(cast<float>(lhs), cast<float>(rhs)));    
            }
            if constexpr (N > 1) {
                return join(
                    mul(lhs.lo, rhs.lo),
                    mul(lhs.hi, rhs.hi)
                );
            }
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return from_vec<T>(vmul_s8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return from_vec<T>(vmulq_s8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return from_vec<T>(vmul_s16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return from_vec<T>(vmulq_s16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return from_vec<T>(vmul_s32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return from_vec<T>(vmulq_s32(to_vec(lhs), to_vec(rhs)));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return from_vec<T>(vmul_u8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return from_vec<T>(vmulq_u8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return from_vec<T>(vmul_u16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return from_vec<T>(vmulq_u16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return from_vec<T>(vmul_u32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return from_vec<T>(vmulq_u32(to_vec(lhs), to_vec(rhs)));
                }
            }
            if constexpr (N > 1) {
                return join(
                    mul(lhs.lo, rhs.lo),
                    mul(lhs.hi, rhs.hi)
                );
            }
        }
    }

// !MARK

// MARK: Extended Multiplication

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto safe_mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                return from_vec<T>(vmulx_f64(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, float>) {
                return {
                    .val = vmulxs_f32(lhs.val, rhs.val)
                };
            }
            #endif
            using std::fpclassify;
            using std::signbit;
            auto lc = fpclassify(lhs.val);
            auto rc = fpclassify(rhs.val);
            auto linf = lc == FP_INFINITE;
            auto rinf = rc == FP_INFINITE;
            auto lz = lc == FP_ZERO;
            auto rz = rc == FP_ZERO;
            auto ls = signbit(lhs.val);
            auto rs = signbit(rhs.val);
            auto sign = ls || rs;

            if ((linf && rz) || (lz && rinf)) {
                return { .val = static_cast<T>(std::copysign<T>(2.0, (sign ? -1 : 1))) };
            } else if (lz || rz) {
                return { .val = static_cast<T>(std::copysign<T>(0.0, (sign ? -1 : 1))) };
            } else if (linf || rinf) {
                return { .val = static_cast<T>(std::copysign<T>(INFINITY, (sign ? -1 : 1))) };
            }
            return { .val = static_cast<T>(lhs.val * rhs.val) };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return from_vec<T>(vmulx_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4)
                    return from_vec<T>(vmulxq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2)
                    return from_vec<T>(vmulxq_f64(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vmulx_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vmulxq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(safe_mul(cast<float>(lhs), cast<float>(rhs)));    
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(safe_mul(cast<float>(lhs), cast<float>(rhs)));    
            }
            #endif

            return join(
                safe_mul(lhs.lo, rhs.lo),
                safe_mul(lhs.hi, rhs.hi)
            );
        }
    }

    // INFO: for integral types, it's same as calling `mul`
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto safe_mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return mul(lhs, rhs);
    }


    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto safe_mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;
        #ifdef UI_CPU_ARM64
        if constexpr (N == 1 && (std::same_as<T, float> || std::same_as<T, float16>)) {
        #else
        if constexpr (true) {
        #endif
            return safe_mul(a, Vec<1, T>(v[Lane]));
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (M == 1) {
                    return safe_mul(a, ret_t::load(v.val));
                } else if constexpr (M <= 4) {
                    if constexpr (N == 2) {
                        if constexpr (M == 2) {
                            return from_vec<T>(
                                vmulx_lane_f32(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return from_vec<T>(
                                vmulx_laneq_f32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 4) {
                        if constexpr (M == 2) {
                            return from_vec<T>(
                                vmulxq_lane_f32(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return from_vec<T>(
                                vmulxq_laneq_f32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    }
                    return join(
                        safe_mul<Lane>(a.lo, v),
                        safe_mul<Lane>(a.hi, v)
                    );
                } else {
                    if constexpr (Lane < M / 2) {
                        return safe_mul<Lane>(a, v.lo);
                    } else {
                        return safe_mul<Lane - M / 2>(a, v.hi);
                    }
                }
                #else
                    return join(
                        safe_mul<Lane>(a.lo, v),
                        safe_mul<Lane>(a.hi, v)
                    );
                #endif
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (M <= 2) {
                    if constexpr (N == 1) {
                        if constexpr (M == 1) {
                            return from_vec<T>(
                                vmulx_lane_f64(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return from_vec<T>(
                                vmulx_laneq_f64(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 2) {
                        if constexpr (M == 1) {
                            return from_vec<T>(
                                vmulxq_lane_f64(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return from_vec<T>(
                                vmulxq_laneq_f64(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else {
                        return join(
                            safe_mul<Lane>(a.lo, v),
                            safe_mul<Lane>(a.hi, v)
                        );
                    }
                } else {
                    if constexpr (Lane < M / 2) {
                        return safe_mul<Lane>(a, v.lo);
                    } else {
                        return safe_mul<Lane - M / 2>(a, v.hi);
                    }
                }
                #else
                    return join(
                        safe_mul<Lane>(a.lo, v),
                        safe_mul<Lane>(a.hi, v)
                    );
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmulx_lane_f16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulxq_lane_f16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (M > 8) {
                        if constexpr (Lane < M / 2) {
                            return safe_mul<Lane>(a, v.lo);
                        } else {
                            return safe_mul<Lane - M / 2>(a, v.hi);
                        }
                    }
                    #ifdef UI_CPU_ARM64
                    if constexpr (M == 1) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmulx_n_f16(to_vec(a), v[Lane])
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulxq_n_f16(to_vec(a), v[Lane])
                            );
                        }
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmulx_laneq_f16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulxq_laneq_f16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    }
                    #else
                        return join(
                            safe_mul<Lane>(a.lo, v),
                            safe_mul<Lane>(a.hi, v)
                        );
                    #endif
                #else
                    return cast<T>(safe_mul<Lane>(cast<float>(a), cast<float>(v)));    
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(safe_mul<Lane>(cast<float>(a), cast<float>(v)));    
            }
        }
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::integral T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto safe_mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        return safe_mul(a, Vec<N, T>::load(v[Lane]));
    }

// !MARK

// MARK: Multiply-Accumulate
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(
                        vmla_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            #endif
            return {
                .val = static_cast<T>(acc.val + (lhs.val * rhs.val))
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vmla_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                } else if constexpr (N == 4) {
                    return from_vec<T>(
                        vmlaq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vmlaq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                return cast<T>(mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vmla_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vmlaq_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmla_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlaq_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmla_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlaq_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    return map([](auto a, auto l, auto r){ return a + (l * r); }, acc, lhs, rhs);
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vmla_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vmlaq_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmla_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlaq_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmla_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlaq_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    return map([](auto a, auto l, auto r){ return a + (l * r); }, acc, lhs, rhs);
                }
            }

            return join(
                mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(
                        vmls_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            #endif
            return {
                .val = static_cast<T>(acc.val - (lhs.val * rhs.val))
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vmls_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                } else if constexpr (N == 4) {
                    return from_vec<T>(
                        vmlsq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vmlsq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                return cast<T>(mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vmls_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vmlsq_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmls_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlsq_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmls_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlsq_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vmls_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vmlsq_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmls_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlsq_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmls_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlsq_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }

            return join(
                mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;

        if constexpr (N == 1) {
            return { .val = static_cast<result_t>(acc.val + static_cast<result_t>(lhs.val) * static_cast<result_t>(rhs.val)) };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vmlal_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlal_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlal_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vmlal_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlal_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlal_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }

            return join(
                mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;

        if constexpr (N == 1) {
            return { .val = static_cast<result_t>(acc.val - static_cast<result_t>(lhs.val) * static_cast<result_t>(rhs.val)) };
        } else {

            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vmlsl_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlsl_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlsl_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vmlsl_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlsl_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlsl_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }

            return join(
                mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
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
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                return from_vec<T>(
                    vfma_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                );
            }
            #endif
            return {
                .val = acc.val + (lhs.val * rhs.val)
            }; 
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vfma_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                } else if constexpr (N == 4) {
                    return from_vec<T>(
                        vfmaq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vfmaq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vfma_f16(to_vec(acc), to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vfmaq_f16(to_vec(acc), to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
            }

            return join(
                fused_mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                fused_mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                return from_vec<T>(
                    vfms_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                );
            }
            #endif
            return {
                .val = acc.val - (lhs.val * rhs.val)
            }; 
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vfms_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                } else if constexpr (N == 4) {
                    return from_vec<T>(
                        vfmsq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vfmsq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vfms_f16(to_vec(acc), to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vfmsq_f16(to_vec(acc), to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(lhs), cast<float>(rhs), op));
            }

            return join(
                fused_mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                fused_mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                if constexpr (M == 1) {
                    return from_vec<T>(
                        vfma_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                    );
                } else if constexpr (M == 2) {
                    return from_vec<T>(
                        vfma_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                    );
                } else if constexpr (M > 2) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
            }
            #endif
            return fused_mul_acc(acc, a, Vec<N, T>::load(v[Lane]));
        } else {
            if constexpr (std::same_as<T, float>) {
            #ifdef UI_CPU_ARM64
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfma_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vfmaq_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfma_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vfmaq_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M > 4) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
            #endif
            } else if constexpr (std::same_as<T, double>) {
            #ifdef UI_CPU_ARM64
                if constexpr (M == 1) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfmaq_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } 
                } else if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfmaq_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } 
                } else if constexpr (M > 2) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #if defined(UI_HAS_FLOAT_16) && defined(UI_CPU_ARM64)
                if constexpr (M == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vfma_lane_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vfmaq_lane_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M == 8) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vfma_laneq_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vfmaq_laneq_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M > 8) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
                #else
                return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(a), cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(fused_mul_acc<Lane>(cast<float>(acc), cast<float>(a), cast<float>(v), op));
            }

            return join(
                fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                fused_mul_acc<Lane>(acc.hi, a.hi, v, op)
            );
        }
    }
    
    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& acc,
        Vec<N, T> const& a,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                if constexpr (M == 1) {
                    return from_vec<T>(
                        vfms_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                    );
                } else if constexpr (M == 2) {
                    return from_vec<T>(
                        vfms_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                    );
                } else if constexpr (M > 2) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
            }
            #endif
            return fused_mul_acc(acc, a, Vec<N, T>::load(v[Lane]));
        } else {
            if constexpr (std::same_as<T, float>) {
            #ifdef UI_CPU_ARM64
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfms_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vfmsq_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfms_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vfmsq_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M > 4) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
            #endif
            } else if constexpr (std::same_as<T, double>) {
            #ifdef UI_CPU_ARM64
                if constexpr (M == 1) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfmsq_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } 
                } else if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vfmsq_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } 
                } else if constexpr (M > 2) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #if defined(UI_CPU_ARM64) && defined(UI_HAS_FLOAT_16)
                if constexpr (M == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vfms_lane_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vfmsq_lane_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M == 8) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vfms_laneq_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vfmsq_laneq_f16(to_vec(acc), to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M > 8) {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
                #else
                return cast<T>(fused_mul_acc(cast<float>(acc), cast<float>(a), cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(fused_mul_acc<Lane>(cast<float>(acc), cast<float>(a), cast<float>(v), op));
            }

            return join(
                fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                fused_mul_acc<Lane>(acc.hi, a.hi, v, op)
            );
        }
    }
// !MARK

// MARK: Widening Multiplication
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            auto l = static_cast<result_t>(lhs.val);
            auto r = static_cast<result_t>(rhs.val);
            return {
                .val = l * r
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vmull_s8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmull_s16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmull_s32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<result_t>(
                            vmull_u8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmull_u16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmull_u32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            }

            return join(
                widening_mul(lhs.lo, rhs.lo),
                widening_mul(lhs.hi, rhs.hi)
            );
        }
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
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(a.val + b.val * v[Lane])
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmla_lane_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlaq_lane_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmla_laneq_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlaq_laneq_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    }
                #endif
                } else if constexpr (M > 4) {
                    if constexpr (Lane * 2 >= M) {
                        return mul_acc<Lane - M / 2>(a, b, v.hi);
                    } else {
                        return mul_acc<Lane>(a, b, v.lo);
                    }
                }
            } else if constexpr (std::same_as<T, bfloat16> || std::same_as<T, float16>) {
                return cast<T>(mul_acc<Lane>(cast<float>(a), cast<float>(b), cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmla_lane_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlaq_lane_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmla_laneq_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlaq_laneq_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmla_lane_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlaq_lane_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmla_laneq_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlaq_laneq_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmla_lane_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlaq_lane_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmla_laneq_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlaq_laneq_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmla_lane_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlaq_lane_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmla_laneq_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlaq_laneq_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                }
            }
            return join(
                mul_acc<Lane>(a.lo, b.lo, v, op),
                mul_acc<Lane>(a.hi, b.hi, v, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(a.val + b.val * c)
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vmla_n_f32(to_vec(a), to_vec(b), c)
                    );
                } else if constexpr (N == 4) {
                    return from_vec<T>(
                        vmlaq_n_f32(to_vec(a), to_vec(b), c)
                    );
                }
            } else if constexpr (std::same_as<T, bfloat16> || std::same_as<T, float16>) {
                return cast<T>(mul_acc(cast<float>(a), cast<float>(b), c, op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmla_n_s16(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlaq_n_s16(to_vec(a), to_vec(b), c)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmla_n_s32(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlaq_n_s32(to_vec(a), to_vec(b), c)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmla_n_u16(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlaq_n_u16(to_vec(a), to_vec(b), c)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmla_n_u32(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlaq_n_u32(to_vec(a), to_vec(b), c)
                        );
                    }
                }
            }
            return join(
                mul_acc(a.lo, b.lo, c, op),
                mul_acc(a.hi, b.hi, c, op)
            );
        }
    }
// !MARK

// MARK: Vector multiply-subtract by scalar
    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        Vec<M, T> const& v,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(a.val - (b.val * v[Lane]))
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmls_lane_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlsq_lane_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmls_laneq_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlsq_laneq_f32(to_vec(a), to_vec(b), to_vec(v), Lane)
                        );
                    }
                #endif
                } else if constexpr (M > 4) {
                    if constexpr (Lane * 2 >= M) {
                        return mul_acc<Lane - M / 2>(a, b, v.hi);
                    } else {
                        return mul_acc<Lane>(a, b, v.lo);
                    }
                }
            } else if constexpr (std::same_as<T, bfloat16> || std::same_as<T, float16>) {
                return cast<T>(mul_acc(cast<float>(a), cast<float>(b), cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmls_lane_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlsq_lane_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmls_laneq_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlsq_laneq_s16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmls_lane_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlsq_lane_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmls_laneq_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlsq_laneq_s32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmls_lane_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlsq_lane_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmls_laneq_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmlsq_laneq_u16(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmls_lane_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlsq_lane_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmls_laneq_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmlsq_laneq_u32(to_vec(a), to_vec(b), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return mul_acc<Lane - M / 2>(a, b, v.hi);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo);
                        }
                    }
                }
            }
            return join(
                mul_acc<Lane>(a.lo, b.lo, v, op),
                mul_acc<Lane>(a.hi, b.hi, v, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(a.val - (b.val * c))
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vmls_n_f32(to_vec(a), to_vec(b), c)
                    );
                } else if constexpr (N == 4) {
                    return from_vec<T>(
                        vmlsq_n_f32(to_vec(a), to_vec(b), c)
                    );
                }
            } else if constexpr (std::same_as<T, bfloat16> || std::same_as<T, float16>) {
                return cast<T>(mul_acc(cast<float>(a), cast<float>(b), c, op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmls_n_s16(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlsq_n_s16(to_vec(a), to_vec(b), c)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmls_n_s32(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlsq_n_s32(to_vec(a), to_vec(b), c)
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vmls_n_u16(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vmlsq_n_u16(to_vec(a), to_vec(b), c)
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmls_n_u32(to_vec(a), to_vec(b), c)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmlsq_n_u32(to_vec(a), to_vec(b), c)
                        );
                    }
                }
            }
            return join(
                mul_acc(a.lo, b.lo, c, op),
                mul_acc(a.hi, b.hi, c, op)
            );
        }
    }
// !MARK

// MARK: Multiplication with scalar
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(v.val * c)
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vmul_n_f32(to_vec(v), c));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vmulq_n_f32(to_vec(v), c));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vmulq_n_f64(to_vec(v), c));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vmul_n_f16(to_vec(v), c));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vmulq_n_f16(to_vec(v), c));
                }
                #else
                return cast<T>(mul_acc(cast<float>(v), c));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(mul_acc(cast<float>(v), c));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vmul_n_s16(to_vec(v), c));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vmulq_n_s16(to_vec(v), c));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vmul_n_s32(to_vec(v), c));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vmulq_n_s32(to_vec(v), c));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vmul_n_u16(to_vec(v), c));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vmulq_n_u16(to_vec(v), c));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vmul_n_u32(to_vec(v), c));
                    } else if constexpr (N == 4) {
                        return from_vec<T>(vmulq_n_u32(to_vec(v), c));
                    }
                }
            }

            return join(
                mul(v.lo, c),
                mul(v.hi, c)
            );
        }
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = static_cast<T>(a.val * v[Lane])
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmul_lane_f32(to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmulq_lane_f32(to_vec(a), to_vec(v), Lane)
                        );
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmul_laneq_f32(to_vec(a), to_vec(v), Lane)
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vmulq_laneq_f32(to_vec(a), to_vec(v), Lane)
                        );
                    }
                #endif
                } else if constexpr (M > 4) {
                    if constexpr (Lane * 2 >= M) {
                        return mul<Lane - M / 2>(a, v.hi);
                    } else {
                        return mul<Lane>(a, v.lo);
                    }
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmul_lane_f16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulq_lane_f16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmul_laneq_f16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulq_laneq_f16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return mul<Lane>(a, v.lo);
                        }
                    }
                #else
                return cast<T>(mul_acc<Lane>(cast<float>(a), cast<float>(v)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(mul_acc<Lane>(cast<float>(a), cast<float>(v)));
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (M == 1) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmulq_lane_f64(to_vec(a), to_vec(v), Lane)
                        );
                    }
                } else if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vmulq_laneq_f64(to_vec(a), to_vec(v), Lane)
                        );
                    }
                } 
            #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmul_lane_s16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulq_lane_s16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmul_laneq_s16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulq_laneq_s16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return mul<Lane>(a, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmul_lane_s32(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmulq_lane_s32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmul_laneq_s32(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmulq_laneq_s32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return mul<Lane>(a, v.lo);
                        }
                    }
                }
            } else {
                if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmul_lane_u16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulq_lane_u16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmul_laneq_u16(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 8) {
                            return from_vec<T>(
                                vmulq_laneq_u16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return mul<Lane>(a, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmul_lane_u32(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmulq_lane_u32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmul_laneq_u32(to_vec(a), to_vec(v), Lane)
                            );
                        } else if constexpr (N == 4) {
                            return from_vec<T>(
                                vmulq_laneq_u32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return mul<Lane>(a, v.lo);
                        }
                    }
                }
            }
            return join(
                mul<Lane>(a.lo, v),
                mul<Lane>(a.hi, v)
            );
        }
    }
// !MARK

// MARK: Multiplication with scalar and widen
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return {
                .val = static_cast<result_t>(static_cast<result_t>(v.val) * static_cast<result_t>(c))
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::int16_t>(v);
                    return mul(temp, static_cast<std::int16_t>(c));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vmull_n_s16(to_vec(v), c));
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vmull_n_s32(to_vec(v), c));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    auto temp = cast<double>(v);
                    return mul(temp, static_cast<double>(c));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::uint16_t>(v);
                    return mul(temp, static_cast<std::int16_t>(c));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vmull_n_u16(to_vec(v), c));
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(vmull_n_u32(to_vec(v), c));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    auto temp = cast<double>(v);
                    return mul(temp, static_cast<double>(c));
                }
            } 
            return join(
                widening_mul(v.lo, c),
                widening_mul(v.hi, c)
            );
        }
    }

    template <unsigned Lane, std::size_t N, std::size_t M, typename T>
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return {
                .val = static_cast<result_t>(static_cast<result_t>(a.val) * static_cast<result_t>(v[Lane]))
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    auto at = cast<std::int16_t>(a);
                    auto vt = cast<std::int16_t>(v);
                    return mul<Lane>(at, vt);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmull_lane_s16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmull_laneq_s16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return widening_mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return widening_mul<Lane>(a, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmull_lane_s32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmull_laneq_s32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return widening_mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return widening_mul<Lane>(a, v.lo);
                        }
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    auto at = cast<std::uint16_t>(a);
                    auto vt = cast<std::uint16_t>(v);
                    return mul<Lane>(at, vt);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmull_lane_u16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<T>(
                                vmull_laneq_u16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 8) {
                        if constexpr (Lane * 2 >= M) {
                            return widening_mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return widening_mul<Lane>(a, v.lo);
                        }
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmull_lane_u32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<T>(
                                vmull_laneq_u32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #endif
                    } else if constexpr (M > 4) {
                        if constexpr (Lane * 2 >= M) {
                            return widening_mul<Lane - M / 2>(a, v.hi);
                        } else {
                            return widening_mul<Lane>(a, v.lo);
                        }
                    }
                }
            } 
            return join(
                widening_mul<Lane>(a.lo, v),
                widening_mul<Lane>(a.hi, v)
            );
        }
    }

// !MARK

// MARK: Vector multiply-accumulate by scalar and widen
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return {
                .val = static_cast<result_t>(a.val + static_cast<result_t>(v.val) * static_cast<result_t>(c))
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::int16_t>(v);
                    return mul_acc(a, temp, static_cast<std::int16_t>(c), op);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlal_n_s16(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlal_n_s32(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 8) {
                    auto temp = cast<double>(v);
                    return mul_acc(a, temp, static_cast<double>(c), op);
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::uint16_t>(v);
                    return mul_acc(a, temp, static_cast<std::int16_t>(c), op);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlal_n_u16(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlal_n_u32(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 8) {
                    auto temp = cast<double>(v);
                    return mul_acc(a, temp, static_cast<double>(c), op);
                }
            } 
            return join(
                widening_mul_acc(a.lo, v.lo, c, op),
                widening_mul_acc(a.hi, v.hi, c, op)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return {
                .val = static_cast<result_t>(a.val + static_cast<result_t>(v.val) * static_cast<result_t>(c))
            };
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::int16_t>(v);
                    return mul_acc(a, temp, static_cast<std::int16_t>(c), op);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlsl_n_s16(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlsl_n_s32(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 8) {
                    auto temp = cast<double>(v);
                    return mul_acc(a, temp, static_cast<double>(c), op);
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::uint16_t>(v);
                    return mul_acc(a, temp, static_cast<std::int16_t>(c), op);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(
                            vmlsl_n_u16(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(
                            vmlsl_n_u32(to_vec(a), to_vec(v), c)
                        );
                    } 
                } else if constexpr (sizeof(T) == 8) {
                    auto temp = cast<double>(v);
                    return mul_acc(a, temp, static_cast<double>(c), op);
                }
            } 
            return join(
                widening_mul_acc(a.lo, v.lo, c, op),
                widening_mul_acc(a.hi, v.hi, c, op)
            );
        }
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
        if constexpr (N == 1) {
            return {
                .val = a.val + (b.val * c)
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vfma_n_f32(to_vec(a), to_vec(b), c));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vfmaq_n_f32(to_vec(a), to_vec(b), c));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vfmaq_n_f64(to_vec(a), to_vec(b), c));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vfma_n_f16(to_vec(a), to_vec(b), c));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vfmaq_n_f16(to_vec(a), to_vec(b), c));
                }
                #else
                return cast<T>(fused_mul_acc(cast<float>(a), cast<float>(b), c, op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(fused_mul_acc(cast<float>(a), cast<float>(b), c, op));
            }
            return join(
                fused_mul_acc(a.lo, b.lo, c, op),
                fused_mul_acc(a.hi, b.hi, c, op)
            );
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        Vec<N, T> const& a,
        Vec<N, T> const& b,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return {
                .val = a.val - (b.val * c)
            };
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vfms_n_f32(to_vec(a), to_vec(b), c));
                } else if constexpr (N == 4) {
                    return from_vec<T>(vfmsq_n_f32(to_vec(a), to_vec(b), c));
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(vfmsq_n_f64(to_vec(a), to_vec(b), c));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vfms_n_f16(to_vec(a), to_vec(b), c));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vfmsq_n_f16(to_vec(a), to_vec(b), c));
                }
                #else
                return cast<T>(fused_mul_acc(cast<float>(a), cast<float>(b), c, op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(fused_mul_acc(cast<float>(a), cast<float>(b), c, op));
            }
            #endif
            return join(
                fused_mul_acc(a.lo, b.lo, c, op),
                fused_mul_acc(a.hi, b.hi, c, op)
            );
        }
    }
// !MARK

} // namespace ui::arm::neon

#endif // AMT_UI_ARCH_ARM_MUL_HPP

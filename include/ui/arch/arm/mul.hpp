#ifndef AMT_UI_ARCH_ARM_MUL_HPP
#define AMT_UI_ARCH_ARM_MUL_HPP

#include "cast.hpp"
#include "basic.hpp"
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <type_traits>

namespace ui::arm {
// MARK: Multiplication

 template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul(
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        if constexpr (N == 1) return { .val = static_cast<T>(lhs.val * rhs.val) };
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vmul_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4) 
                    return std::bit_cast<ret_t>(vmulq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmulq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
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
                        return std::bit_cast<ret_t>(vmul_s8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return std::bit_cast<ret_t>(vmulq_s8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmul_s16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmulq_s16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmul_s32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmulq_s32(to_vec(lhs), to_vec(rhs)));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmul_u8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return std::bit_cast<ret_t>(vmulq_u8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmul_u16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmulq_u16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmul_u32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmulq_u32(to_vec(lhs), to_vec(rhs)));
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
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                return std::bit_cast<ret_t>(vmulx_f64(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, float>) {
                return {
                    .val = vmulxs_f32(lhs.val, rhs.val)
                };
            }
            #endif
            auto lc = std::fpclassify(lhs.val);
            auto rc = std::fpclassify(rhs.val);
            auto linf = lc == FP_INFINITE;
            auto rinf = rc == FP_INFINITE;
            auto lz = lc == FP_ZERO;
            auto rz = rc == FP_ZERO;
            auto ls = std::signbit(lhs.val);
            auto rs = std::signbit(rhs.val);
            auto sign = ls || rs;

            if ((linf && rz) || (lz &&rinf)) {
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
                    return std::bit_cast<ret_t>(vmulx_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4)
                    return std::bit_cast<ret_t>(vmulxq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vmulxq_f64(to_vec(lhs), to_vec(rhs)));
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
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        return mul(lhs, rhs);
    }


    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto safe_mul(
        VecReg<N, T> const& a,
        VecReg<M, T> const& v
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        #ifdef UI_CPU_ARM64
        if constexpr (N == 1 && std::same_as<T, float>) {
        #else
        if constexpr (true) {
        #endif
            return safe_mul(a, VecReg<1, T>(v[Lane]));
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (M == 1) {
                    return safe_mul(a, load<N>(v.val));
                } else if constexpr (M <= 4) {
                    if constexpr (N == 2) {
                        if constexpr (M == 2) {
                            return std::bit_cast<ret_t>(
                                vmulx_lane_f32(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vmulx_laneq_f32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 4) {
                        if constexpr (M == 2) {
                            return std::bit_cast<ret_t>(
                                vmulxq_lane_f32(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
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
                            return std::bit_cast<ret_t>(
                                vmulx_lane_f64(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vmulx_laneq_f64(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 2) {
                        if constexpr (M == 1) {
                            return std::bit_cast<ret_t>(
                                vmulxq_lane_f64(to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
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
            }
    
        }
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::integral T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto safe_mul(
        VecReg<N, T> const& a,
        VecReg<M, T> const& v
    ) noexcept -> VecReg<N, T> {
        return safe_mul(a, load<N>(v[Lane]));
    }

// !MARK

// MARK: Multiply-Accumulate
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs,
        [[maybe_unused]] std::plus<> op
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return std::bit_cast<ret_t>(
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
                    return std::bit_cast<ret_t>(
                        vmla_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(
                        vmlaq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vmlaq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmla_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(
                            vmlaq_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmla_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlaq_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmla_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmlaq_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 8) {
                    return map([](auto a, auto l, auto r){ return a + (l * r); }, acc, lhs, rhs);
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmla_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(
                            vmlaq_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmla_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlaq_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmla_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
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
        VecReg<N, T> const& acc,
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs,
        [[maybe_unused]] std::minus<> op
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return std::bit_cast<ret_t>(
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
                    return std::bit_cast<ret_t>(
                        vmls_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(
                        vmlsq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vmlsq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))
                    );
                }
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmls_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(
                            vmlsq_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmls_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlsq_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmls_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmlsq_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmls_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 16) {
                        return std::bit_cast<ret_t>(
                            vmlsq_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmls_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlsq_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmls_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))
                        );
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
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
        VecReg<N, internal::widening_result_t<T>> const& acc,
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs,
        [[maybe_unused]] std::plus<> op
    ) noexcept -> VecReg<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        using ret_t = VecReg<N, result_t>;

        if constexpr (N == 1) {
            return { .val = static_cast<result_t>(acc.val + static_cast<result_t>(lhs.val) * static_cast<result_t>(rhs.val)) };
        } else {

            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlal_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmlal_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmlal_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlal_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmlal_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmlal_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                }
            }

            return join(
                mul_acc(acc.lo, lhs.lo, rhs.lo),
                mul_acc(acc.hi, lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto mul_acc(
        VecReg<N, internal::widening_result_t<T>> const& acc,
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs,
        [[maybe_unused]] std::minus<> op
    ) noexcept -> VecReg<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        using ret_t = VecReg<N, result_t>;

        if constexpr (N == 1) {
            return { .val = static_cast<result_t>(acc.val - static_cast<result_t>(lhs.val) * static_cast<result_t>(rhs.val)) };
        } else {

            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlsl_s8(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmlsl_s16(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmlsl_s32(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmlsl_u8(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmlsl_u16(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmlsl_u32(to_vec(acc), to_vec(lhs), to_vec(rhs))             
                        );
                    }
                }
            }

            return join(
                mul_acc(acc.lo, lhs.lo, rhs.lo),
                mul_acc(acc.hi, lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, typename T, typename U>
    UI_ALWAYS_INLINE auto mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, U> const& lhs,
        VecReg<N, U> const& rhs
    ) noexcept -> VecReg<N, U> {
        return mul_acc(acc, lhs, rhs, std::plus<>{});
    }

// !MARK

// MARK: Fused-Multiply-Accumulate
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs,
        [[maybe_unused]] std::plus<> op
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                return std::bit_cast<ret_t>(
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
                    return std::bit_cast<ret_t>(
                        vfma_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))        
                    );
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(
                        vfmaq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))        
                    );
                }
            } else if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vfmaq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))        
                    );
                }
                #endif
            }

            return join(
                fused_mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                fused_mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs,
        [[maybe_unused]] std::minus<> op
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                return std::bit_cast<ret_t>(
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
                    return std::bit_cast<ret_t>(
                        vfms_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))        
                    );
                } else if constexpr (N == 4) {
                    return std::bit_cast<ret_t>(
                        vfmsq_f32(to_vec(acc), to_vec(lhs), to_vec(rhs))        
                    );
                }
            } else if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vfmsq_f64(to_vec(acc), to_vec(lhs), to_vec(rhs))        
                    );
                }
                #endif
            }

            return join(
                fused_mul_acc(acc.lo, lhs.lo, rhs.lo, op),
                fused_mul_acc(acc.hi, lhs.hi, rhs.hi, op)
            );
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fused_mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        return fused_mul_acc(acc, lhs, rhs, std::plus<>{});
    }


    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, T> const& a,
        VecReg<M, T> const& v,
        [[maybe_unused]] std::plus<> op
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        #ifdef UI_CPU_ARM64
        if constexpr (N == 1 && std::same_as<T, float>) {
        #else
        if constexpr (true) {
        #endif
            return fused_mul_acc(acc, a, VecReg<1, T>(v[Lane]));
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (M == 1) {
                    return safe_mul(a, load<N>(v.val));
                } else if constexpr (M <= 4) {
                    if constexpr (N == 2) {
                        if constexpr (M == 2) {
                            return std::bit_cast<ret_t>(
                                vfma_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfma_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 4) {
                        if constexpr (M == 2) {
                            return std::bit_cast<ret_t>(
                                vfmaq_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfmaq_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    }
                    return join(
                        fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                        fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                    );
                } else {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
                #else
                    return join(
                        fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                        fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                    );
                #endif
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (M <= 2) {
                    if constexpr (N == 1) {
                        if constexpr (M == 1) {
                            return std::bit_cast<ret_t>(
                                vfma_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfma_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 2) {
                        if constexpr (M == 1) {
                            return std::bit_cast<ret_t>(
                                vfmaq_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfmaq_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else {
                        return join(
                            fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                            fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                        );
                    }
                } else {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
                #else
                    return join(
                        fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                        fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                    );
                #endif
            }
    
        }
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, T> const& a,
        VecReg<M, T> const& v,
        [[maybe_unused]] std::minus<> op
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        #ifdef UI_CPU_ARM64
        if constexpr (N == 1 && std::same_as<T, float>) {
        #else
        if constexpr (true) {
        #endif
            return fused_mul_acc(acc, a, VecReg<1, T>(v[Lane]), op);
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (M == 1) {
                    return safe_mul(a, load<N>(v.val));
                } else if constexpr (M <= 4) {
                    if constexpr (N == 2) {
                        if constexpr (M == 2) {
                            return std::bit_cast<ret_t>(
                                vfms_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfms_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 4) {
                        if constexpr (M == 2) {
                            return std::bit_cast<ret_t>(
                                vfmsq_lane_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfmsq_laneq_f32(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    }
                    return join(
                        fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                        fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                    );
                } else {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
                #else
                    return join(
                        fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                        fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                    );
                #endif
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (M <= 2) {
                    if constexpr (N == 1) {
                        if constexpr (M == 1) {
                            return std::bit_cast<ret_t>(
                                vfms_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfms_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else if constexpr (N == 2) {
                        if constexpr (M == 1) {
                            return std::bit_cast<ret_t>(
                                vfmsq_lane_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        } else {
                            return std::bit_cast<ret_t>(
                                vfmsq_laneq_f64(to_vec(acc), to_vec(a), to_vec(v), Lane)
                            );
                        }
                    } else {
                        return join(
                            fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                            fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                        );
                    }
                } else {
                    if constexpr (Lane < M / 2) {
                        return fused_mul_acc<Lane>(acc, a, v.lo, op);
                    } else {
                        return fused_mul_acc<Lane - M / 2>(acc, a, v.hi, op);
                    }
                }
                #else
                    return join(
                        fused_mul_acc<Lane>(acc.lo, a.lo, v, op),
                        fused_mul_acc<Lane>(acc.lo, a.hi, v, op)
                    );
                #endif
            }
    
        }
    }

    template <std::size_t Lane, std::size_t N, std::size_t M, std::floating_point T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto fused_mul_acc(
        VecReg<N, T> const& acc,
        VecReg<N, T> const& a,
        VecReg<M, T> const& v
    ) noexcept -> VecReg<N, T> {
        return fused_mul_acc<Lane>(acc, a, v, std::plus<>{});
    }

// !MARK

// MARK: Widening Multiplication
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto widening_mul(
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        using ret_t = VecReg<N, result_t>;
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
                        return std::bit_cast<ret_t>(
                            vmull_s8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmull_s16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vmull_s32(to_vec(lhs), to_vec(rhs))
                        );
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return std::bit_cast<ret_t>(
                            vmull_u8(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vmull_u16(to_vec(lhs), to_vec(rhs))
                        );
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
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

} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_MUL_HPP

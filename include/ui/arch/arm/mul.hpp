#ifndef AMT_UI_ARCH_ARM_MUL_HPP
#define AMT_UI_ARCH_ARM_MUL_HPP

#include "cast.hpp"
#include "../emul/mul.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
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
            return emul::mul_acc(acc, lhs, rhs, op);
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
            return emul::mul_acc(acc, lhs, rhs, op);
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

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template <std::size_t N, std::integral T>
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;

        if constexpr (N == 1) {
            return emul::mul_acc(acc, lhs, rhs, op);
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
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto mul_acc(
        Vec<N, internal::widening_result_t<T>> const& acc,
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs,
        op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;

        if constexpr (N == 1) {
            return emul::mul_acc(acc, lhs, rhs, op);
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
            return emul::fused_mul_acc(acc, lhs, rhs, op);
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
            return emul::fused_mul_acc(acc, lhs, rhs, op);
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
            return fused_mul_acc(acc, a, Vec<N, T>::load(v[Lane]), op);
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
            return fused_mul_acc(acc, a, Vec<N, T>::load(v[Lane]), op);
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
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return emul::widening_mul(lhs, rhs);
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
            return emul::mul_acc<Lane>(a, b, v, op);
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
                        return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                    } else {
                        return mul_acc<Lane>(a, b, v.lo, op);
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
            return emul::mul_acc(a, b, c, op);
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
                return cast<T>(mul_acc(cast<float>(a), cast<float>(b), static_cast<float>(c), op));
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
            return emul::mul_acc<Lane>(a, b, v, op);
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
                        return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                    } else {
                        return mul_acc<Lane>(a, b, v.lo, op);
                    }
                }
            } else if constexpr (std::same_as<T, bfloat16> || std::same_as<T, float16>) {
                return cast<T>(mul_acc<Lane>(cast<float>(a), cast<float>(b), cast<float>(v), op));
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
                            return mul_acc<Lane - M / 2>(a, b, v.hi, op);
                        } else {
                            return mul_acc<Lane>(a, b, v.lo, op);
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
            return emul::mul_acc(a, b, c, op);
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
                return cast<T>(mul_acc(cast<float>(a), cast<float>(b), static_cast<float>(c), op));
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
            return emul::mul(v, c);
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
                    return from_vec<T>(vmul_n_f16(to_vec(v), std::bit_cast<float16_t>(c)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vmulq_n_f16(to_vec(v), std::bit_cast<float16_t>(c)));
                }
                #else
                return cast<T>(mul(cast<float>(v), static_cast<float>(c)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(mul(cast<float>(v), static_cast<float>(c)));
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
            return emul::mul<Lane>(a, v);
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
                return cast<T>(mul<Lane>(cast<float>(a), cast<float>(v)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(mul<Lane>(cast<float>(a), cast<float>(v)));
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
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& v,
        T const c
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return emul::widening_mul(v, c);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::int16_t>(v);
                    return mul(temp, static_cast<std::int16_t>(c));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(vmull_n_s16(to_vec(v), c));
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(vmull_n_s32(to_vec(v), c));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    auto temp = cast<double>(v);
                    return mul(temp, static_cast<double>(c));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    auto temp = cast<std::uint16_t>(v);
                    return mul(temp, static_cast<std::uint16_t>(c));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<result_t>(vmull_n_u16(to_vec(v), c));
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<result_t>(vmull_n_u32(to_vec(v), c));
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
        requires (Lane < M && sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul(
        Vec<N, T> const& a,
        Vec<M, T> const& v
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return emul::widening_mul<Lane>(a, v);
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    auto at = cast<std::int16_t>(a);
                    auto vt = cast<std::int16_t>(v);
                    return mul<Lane>(at, vt);
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec<result_t>(
                                vmull_lane_s16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<result_t>(
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
                            return from_vec<result_t>(
                                vmull_lane_s32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<result_t>(
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
                            return from_vec<result_t>(
                                vmull_lane_u16(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec<result_t>(
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
                            return from_vec<result_t>(
                                vmull_lane_u32(to_vec(a), to_vec(v), Lane)
                            );
                        }
                    #ifdef UI_CPU_ARM64
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec<result_t>(
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
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::add_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return emul::widening_mul_acc(a, v, c, op);
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
                    return mul_acc(a, temp, static_cast<std::uint16_t>(c), op);
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
        requires (sizeof(T) < 8)
    UI_ALWAYS_INLINE auto widening_mul_acc(
        Vec<N, internal::widening_result_t<T>> const& a,
        Vec<N, T> const& v,
        T const c,
        op::sub_t op
    ) noexcept -> Vec<N, internal::widening_result_t<T>> {
        using result_t = internal::widening_result_t<T>;
        if constexpr (N == 1) {
            return emul::widening_mul(a, v, c, op);
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
            return emul::fused_mul_acc(a, b, c, op);
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
            return emul::fused_mul_acc(a, b, c, op);
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

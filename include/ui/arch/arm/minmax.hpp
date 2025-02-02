#ifndef AMT_UI_ARCH_ARM_MINMAX_HPP
#define AMT_UI_ARCH_ARM_MINMAX_HPP

#include "cast.hpp"
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <type_traits>

namespace ui::arm {

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto max(
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        if constexpr (N == 1) return { .val = std::max(lhs.val, rhs.val) };
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vmax_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4) 
                    return std::bit_cast<ret_t>(vmaxq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmaxq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            }
            if constexpr (N > 1) {
                return join(
                    max(lhs.lo, rhs.lo),
                    max(lhs.hi, rhs.hi)
                );
            }
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmax_s8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return std::bit_cast<ret_t>(vmaxq_s8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmax_s16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmaxq_s16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmax_s32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmaxq_s32(to_vec(lhs), to_vec(rhs)));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmax_u8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return std::bit_cast<ret_t>(vmaxq_u8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmax_u16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmaxq_u16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmax_u32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmaxq_u32(to_vec(lhs), to_vec(rhs)));
                }
            }
            if constexpr (N > 1) {
                return join(
                    max(lhs.lo, rhs.lo),
                    max(lhs.hi, rhs.hi)
                );
            }
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto min(
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;
        if constexpr (N == 1) return { .val = std::min(lhs.val, rhs.val) };
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vmin_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4) 
                    return std::bit_cast<ret_t>(vminq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vminq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            }
            if constexpr (N > 1) {
                return join(
                    min(lhs.lo, rhs.lo),
                    min(lhs.hi, rhs.hi)
                );
            }
        } else {
            if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmin_s8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return std::bit_cast<ret_t>(vminq_s8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmin_s16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vminq_s16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmin_s32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vminq_s32(to_vec(lhs), to_vec(rhs)));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vmin_u8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return std::bit_cast<ret_t>(vminq_u8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vmin_u16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return std::bit_cast<ret_t>(vminq_u16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return std::bit_cast<ret_t>(vmin_u32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return std::bit_cast<ret_t>(vminq_u32(to_vec(lhs), to_vec(rhs)));
                }
            }
            if constexpr (N > 1) {
                return join(
                    min(lhs.lo, rhs.lo),
                    min(lhs.hi, rhs.hi)
                );
            }
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto maxnm(
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return std::bit_cast<ret_t>(
                        vmaxnm_f64(to_vec(lhs), to_vec(rhs))
                    );
                }
            #endif
            if (std::isnan(lhs.val)) return { .val = rhs.val };
            if (std::isnan(rhs.val)) return { .val = lhs.val };
            return {
                .val = std::max(lhs.val, rhs.val)
            };
        } else {

            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vmaxnm_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4)
                    return std::bit_cast<ret_t>(vmaxnmq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vmaxnmq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            }

            return join(
                maxnm(lhs.lo, rhs.lo),
                maxnm(lhs.hi, rhs.hi)
            );
        }
    }

    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto minnm(
        VecReg<N, T> const& lhs,
        VecReg<N, T> const& rhs
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return std::bit_cast<ret_t>(
                        vminnm_f64(to_vec(lhs), to_vec(rhs))
                    );
                }
            #endif
            if (std::isnan(lhs.val)) return { .val = rhs.val };
            if (std::isnan(rhs.val)) return { .val = lhs.val };
            return {
                .val = std::min(lhs.val, rhs.val)
            };
        } else {

            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vminnm_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4)
                    return std::bit_cast<ret_t>(vminnmq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2)
                    return std::bit_cast<ret_t>(vminnmq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            }

            return join(
                minnm(lhs.lo, rhs.lo),
                minnm(lhs.hi, rhs.hi)
            );
        }
    }
} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_MINMAX_HPP

#ifndef AMT_UI_ARCH_ARM_ROUNDING_HPP
#define AMT_UI_ARCH_ARM_ROUNDING_HPP

#include "cast.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include "basic.hpp"

namespace ui::arm {

    namespace internal {
        // TODO: Implement arm algorithm using FPCR register
        // https://developer.arm.com/documentation/ddi0596/2021-03/Shared-Pseudocode/Shared-Functions?lang=en#impl-shared.FPRoundInt.4
        template <std::floating_point T>
        UI_ALWAYS_INLINE static auto round_helper(T val, std::float_round_style mode) noexcept -> T {
            auto const old = std::fegetround(); 
            std::fesetround(mode);
            auto res = std::round(val);
            std::fesetround(old);
            return res;
        }
    } // namespace internal

    template <std::float_round_style mode = std::float_round_style::round_toward_zero, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto round(
        VecReg<N, T> const& v
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                if constexpr (mode == std::float_round_style::round_toward_zero) {
                    return std::bit_cast<ret_t>(
                        vrnd_f64(to_vec(v))
                    );
                } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                    return std::bit_cast<ret_t>(
                        vrndn_f64(to_vec(v))
                    );
                } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                    return std::bit_cast<ret_t>(
                        vrndp_f64(to_vec(v))
                    );
                } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                    return std::bit_cast<ret_t>(
                        vrndm_f64(to_vec(v))
                    );
                } else {
                    return std::bit_cast<ret_t>(
                        vrndx_f64(to_vec(v))
                    );
                }
            }
            #endif

            return {
                .val = internal::round_helper(v.val, mode)
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return std::bit_cast<ret_t>(
                            vrnd_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return std::bit_cast<ret_t>(
                            vrndn_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return std::bit_cast<ret_t>(
                            vrndp_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return std::bit_cast<ret_t>(
                            vrndm_f32(to_vec(v))
                        );
                    } else {
                        return std::bit_cast<ret_t>(
                            vrndx_f32(to_vec(v))
                        );
                    }

                } else if constexpr (N == 4) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return std::bit_cast<ret_t>(
                            vrndq_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return std::bit_cast<ret_t>(
                            vrndnq_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return std::bit_cast<ret_t>(
                            vrndpq_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return std::bit_cast<ret_t>(
                            vrndmq_f32(to_vec(v))
                        );
                    } else {
                        return std::bit_cast<ret_t>(
                            vrndxq_f32(to_vec(v))
                        );
                    }
                }
            } else if (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return std::bit_cast<ret_t>(
                            vrndq_f64(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return std::bit_cast<ret_t>(
                            vrndnq_f64(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return std::bit_cast<ret_t>(
                            vrndpq_f64(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return std::bit_cast<ret_t>(
                            vrndmq_f64(to_vec(v))
                        );
                    } else {
                        return std::bit_cast<ret_t>(
                            vrndxq_f64(to_vec(v))
                        );
                    }

                }
                #endif
            }    

            return join(
                round<mode>(v.lo),
                round<mode>(v.hi)
            );
        }
    } 
    
} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_ROUNDING_HPP

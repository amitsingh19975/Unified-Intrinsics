#ifndef AMT_UI_ARCH_ARM_ROUNDING_HPP
#define AMT_UI_ARCH_ARM_ROUNDING_HPP

#include "cast.hpp"
#include <cassert>
#include <cfenv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include "../emul/rounding.hpp"

namespace ui::arm::neon {

    namespace internal {
        // TODO: Implement arm algorithm using FPCR register
        // https://developer.arm.com/documentation/ddi0596/2021-03/Shared-Pseudocode/Shared-Functions?lang=en#impl-shared.FPRoundInt.4
        template <std::floating_point T>
        UI_ALWAYS_INLINE static auto round_helper(T val, std::float_round_style mode) noexcept -> T {
            auto const old = std::fegetround(); 
            std::fesetround(::ui::internal::convert_rounding_style(mode));
            auto res = std::nearbyint(val);
            std::fesetround(old);
            return res;
        }
    } // namespace internal

    template <std::float_round_style mode = std::float_round_style::round_to_nearest, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto round(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, double>) {
                if constexpr (mode == std::float_round_style::round_toward_zero) {
                    return from_vec<T>(
                        vrnd_f64(to_vec(v))
                    );
                } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                    return from_vec<T>(
                        vrndn_f64(to_vec(v))
                    );
                } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                    return from_vec<T>(
                        vrndp_f64(to_vec(v))
                    );
                } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                    return from_vec<T>(
                        vrndm_f64(to_vec(v))
                    );
                } else {
                    return from_vec<T>(
                        vrndx_f64(to_vec(v))
                    );
                }
            }
            #endif
            return emul::round<mode>(v);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            vrnd_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            vrndn_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            vrndp_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            vrndm_f32(to_vec(v))
                        );
                    } else {
                        return from_vec<T>(
                            vrndx_f32(to_vec(v))
                        );
                    }

                } else if constexpr (N == 4) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            vrndq_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            vrndnq_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            vrndpq_f32(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            vrndmq_f32(to_vec(v))
                        );
                    } else {
                        return from_vec<T>(
                            vrndxq_f32(to_vec(v))
                        );
                    }
                }
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            vrndq_f64(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            vrndnq_f64(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            vrndpq_f64(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            vrndmq_f64(to_vec(v))
                        );
                    } else {
                        return from_vec<T>(
                            vrndxq_f64(to_vec(v))
                        );
                    }

                }
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            vrnd_f16(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            vrndn_f16(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            vrndp_f16(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            vrndm_f16(to_vec(v))
                        );
                    } else {
                        return from_vec<T>(
                            vrndx_f16(to_vec(v))
                        );
                    }
                } else if constexpr (N == 8) {
                    if constexpr (mode == std::float_round_style::round_toward_zero) {
                        return from_vec<T>(
                            vrndq_f16(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_to_nearest) {
                        return from_vec<T>(
                            vrndnq_f16(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_infinity) {
                        return from_vec<T>(
                            vrndpq_f16(to_vec(v))
                        );
                    } else if constexpr (mode == std::float_round_style::round_toward_neg_infinity) {
                        return from_vec<T>(
                            vrndmq_f16(to_vec(v))
                        );
                    } else {
                        return from_vec<T>(
                            vrndxq_f16(to_vec(v))
                        );
                    }
                }
                #else
                return cast<T>(round<mode>(cast<float>(v))); 
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(round<mode>(cast<float>(v))); 
            }

            return join(
                round<mode>(v.lo),
                round<mode>(v.hi)
            );
        }
    } 
    
} // namespace ui::arm::neon

#endif // AMT_UI_ARCH_ARM_ROUNDING_HPP

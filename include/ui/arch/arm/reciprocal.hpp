#ifndef AMT_UI_ARCH_ARM_RECIPROCAL_HPP
#define AMT_UI_ARCH_ARM_RECIPROCAL_HPP

#include "cast.hpp"
#include <bit>
#include <cassert>
#include <cfenv>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include "../../modular_inv.hpp"

namespace ui::arm {

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto reciprocal_estimate(
        VecReg<N, T> const& v
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = vrecpes_f32(v.val)
                };
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = vrecped_f64(v.val)
                };

            }
            #endif
            return {
                .val = maths::BinaryReciprocal{}.estimate(v.val)
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrecpe_f32(to_vec(v))
                    );
                } else if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrecpeq_f32(to_vec(v))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrecpeq_f64(to_vec(v))
                    );
                }
            #endif
            } else {
                if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vrecpe_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vrecpeq_u32(to_vec(v))
                        );
                    }
                }
            }

            return join(
                reciprocal_estimate(v.lo),
                reciprocal_estimate(v.hi)
            );
        }
    }
    

    /**
     * @param v orginal vector
     * @param e previous esimate
     * @return new estimate
    */
    template <std::size_t N, typename T>
        requires (std::floating_point<T>)
    UI_ALWAYS_INLINE auto reciprocal_refine(
        VecReg<N, T> const& v,
        VecReg<N, T> const& e
    ) noexcept -> VecReg<N, T> {
        using ret_t = VecReg<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = vrecpss_f32(v.val, e.val)
                };
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = vrecpsd_f64(v.val, e.val)
                };

            }
            #endif
            return {
                .val = maths::internal::calculate_reciprocal(v.val, e.val)
            };

        } else {

            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrecps_f32(to_vec(v), to_vec(e))
                    );
                } else if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrecpsq_f32(to_vec(v), to_vec(e))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrecpsq_f64(to_vec(v), to_vec(e))
                    );
                }
            #endif
            }

            return join(
                reciprocal_refine(v.lo, e.lo),
                reciprocal_refine(v.hi, e.hi)
            );
        }
    }
} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_RECIPROCAL_HPP

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
#include "ui/maths.hpp"

namespace ui::arm {

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

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
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto reciprocal_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

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

    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sqrt_inv_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = vrsqrtes_f32(v.val)
                };
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = vrsqrted_f64(v.val)
                };

            }
            #endif
            return {
                .val = maths::BinaryReciprocal{}.sqrt_inv(v.val)
            };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrsqrte_f32(to_vec(v))
                    );
                } else if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrsqrteq_f32(to_vec(v))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrsqrteq_f64(to_vec(v))
                    );
                }
            #endif
            } else {
                if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return std::bit_cast<ret_t>(
                            vrsqrte_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return std::bit_cast<ret_t>(
                            vrsqrteq_u32(to_vec(v))
                        );
                    }
                }
            }

            return join(
                sqrt_inv_estimate(v.lo),
                sqrt_inv_estimate(v.hi)
            );
        }
    }

    /**
     * @param v orginal vector
     * @param e previous esimate
     * @return new estimate
    */
    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sqrt_inv_refine(
        Vec<N, T> const& v,
        Vec<N, T> const& e
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return {
                    .val = vrsqrtss_f32(v.val, e.val)
                };
            } else if constexpr (std::same_as<T, double>) {
                return {
                    .val = vrsqrtsd_f64(v.val, e.val)
                };

            }
            #endif
            return {
                .val = maths::internal::calculate_sqrt_inv(v.val, e.val)
            };

        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrsqrts_f32(to_vec(v), to_vec(e))
                    );
                } else if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrsqrtsq_f32(to_vec(v), to_vec(e))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return std::bit_cast<ret_t>(
                        vrsqrtsq_f64(to_vec(v), to_vec(e))
                    );
                }
            #endif
            }

            return join(
                sqrt_inv_refine(v.lo, e.lo),
                sqrt_inv_refine(v.hi, e.hi)
            );
        }
    }

    template <std::size_t N, typename T>
        requires (std::floating_point<T>)
    UI_ALWAYS_INLINE auto exponent_reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, float>) {
                    return { .val = vrecpxs_f32(v.val) };
                } else if constexpr (std::same_as<T, double>) {
                    return { .val = vrecpxd_f64(v.val) };
                }
            #else
                auto fp = maths::decompose_fp(v.val);
                return reciprocal_estimate(Vec<1, T>(1 << fp.exponent));
            #endif
        } else {
            return join(
                exponent_reciprocal_estimate(v.lo),
                exponent_reciprocal_estimate(v.hi)
            );
        }
    }

} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_RECIPROCAL_HPP

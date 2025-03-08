#ifndef AMT_UI_ARCH_ARM_RECIPROCAL_HPP
#define AMT_UI_ARCH_ARM_RECIPROCAL_HPP

#include "cast.hpp"
#include <concepts>
#include <cstddef>
#include "../emul/reciprocal.hpp"

namespace ui::arm::neon {
    template <std::size_t N, typename T>
        requires (std::floating_point<T> || !std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto reciprocal_estimate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
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
            return emul::reciprocal_estimate(v);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrecpe_f32(to_vec(v))
                    );
                } else if constexpr (N == 2) {
                    return from_vec<T>(
                        vrecpeq_f32(to_vec(v))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrecpeq_f64(to_vec(v))
                    );
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec(vrecpe_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec(vrecpeq_f16(to_vec(v)));
                }
                #else
                return cast<T>(reciprocal_estimate(cast<float>(v)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(reciprocal_estimate(cast<float>(v)));
            } else {
                if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrecpe_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
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
            return emul::reciprocal_refine(v, e);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrecps_f32(to_vec(v), to_vec(e))
                    );
                } else if constexpr (N == 2) {
                    return from_vec<T>(
                        vrecpsq_f32(to_vec(v), to_vec(e))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrecpsq_f64(to_vec(v), to_vec(e))
                    );
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec(vrecps_f16(to_vec(v), to_vec(e)));
                } else if constexpr (N == 8) {
                    return from_vec(vrecpsq_f16(to_vec(v), to_vec(e)));
                }
                #else
                return cast<T>(reciprocal_refine(cast<float>(v), cast<float>(e)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(reciprocal_refine(cast<float>(v), cast<float>(e)));
            } else if constexpr (std::integral<T>) {
                return e;
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
            return emul::sqrt_inv_estimate(v);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrsqrte_f32(to_vec(v))
                    );
                } else if constexpr (N == 2) {
                    return from_vec<T>(
                        vrsqrteq_f32(to_vec(v))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrsqrteq_f64(to_vec(v))
                    );
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec(vrsqrte_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return from_vec(vrsqrteq_f16(to_vec(v)));
                }
                #else
                return cast<T>(sqrt_inv_estimate(cast<float>(v)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(sqrt_inv_estimate(cast<float>(v)));
            } else {
                if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vrsqrte_u32(to_vec(v))
                        );
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
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
            return emul::sqrt_inv_refine(v, e);
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrsqrts_f32(to_vec(v), to_vec(e))
                    );
                } else if constexpr (N == 2) {
                    return from_vec<T>(
                        vrsqrtsq_f32(to_vec(v), to_vec(e))
                    );
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec<T>(
                        vrsqrtsq_f64(to_vec(v), to_vec(e))
                    );
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec(vrsqrts_f16(to_vec(v), to_vec(e)));
                } else if constexpr (N == 8) {
                    return from_vec(vrsqrtsq_f16(to_vec(v), to_vec(e)));
                }
                #else
                return cast<T>(sqrt_inv_refine(cast<float>(v), cast<float>(e)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(sqrt_inv_refine(cast<float>(v), cast<float>(e)));
            } else if constexpr (std::integral<T>) {
                return e;
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
            #endif
            return emul::exponent_reciprocal_estimate(v);
        } else { 
            return join(
                exponent_reciprocal_estimate(v.lo),
                exponent_reciprocal_estimate(v.hi)
            );
        }
    }

} // namespace ui::arm::neon

#endif // AMT_UI_ARCH_ARM_RECIPROCAL_HPP

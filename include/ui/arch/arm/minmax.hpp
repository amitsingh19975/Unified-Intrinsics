#ifndef AMT_UI_ARCH_ARM_MINMAX_HPP
#define AMT_UI_ARCH_ARM_MINMAX_HPP

#include "cast.hpp"
#include "ui/arch/basic.hpp"
#include "ui/base.hpp"
#include "ui/float.hpp"
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <type_traits>

namespace ui::arm::neon {

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto max(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) return { .val = std::max(lhs.val, rhs.val) };
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return from_vec<T>(vmax_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4) 
                    return from_vec<T>(vmaxq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                    if constexpr (N == 2)
                        return from_vec<T>(vmaxq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vmax_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vmaxq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(max(cast<float>(lhs), cast<float>(rhs)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(max(cast<float>(lhs), cast<float>(rhs)));
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
                        return from_vec<T>(vmax_s8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return from_vec<T>(vmaxq_s8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return from_vec<T>(vmax_s16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return from_vec<T>(vmaxq_s16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return from_vec<T>(vmax_s32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return from_vec<T>(vmaxq_s32(to_vec(lhs), to_vec(rhs)));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return from_vec<T>(vmax_u8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return from_vec<T>(vmaxq_u8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return from_vec<T>(vmax_u16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return from_vec<T>(vmaxq_u16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return from_vec<T>(vmax_u32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return from_vec<T>(vmaxq_u32(to_vec(lhs), to_vec(rhs)));
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
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) return { .val = std::min(lhs.val, rhs.val) };
        if constexpr (std::floating_point<T>) {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return from_vec<T>(vmin_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4) 
                    return from_vec<T>(vminq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                    if constexpr (N == 2)
                        return from_vec<T>(vminq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vmin_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vminq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(max(cast<float>(lhs), cast<float>(rhs)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(min(cast<float>(lhs), cast<float>(rhs)));
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
                        return from_vec<T>(vmin_s8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return from_vec<T>(vminq_s8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return from_vec<T>(vmin_s16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return from_vec<T>(vminq_s16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return from_vec<T>(vmin_s32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return from_vec<T>(vminq_s32(to_vec(lhs), to_vec(rhs)));
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8)
                        return from_vec<T>(vmin_u8(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 16)
                        return from_vec<T>(vminq_u8(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4)
                        return from_vec<T>(vmin_u16(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 8)
                        return from_vec<T>(vminq_u16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2)
                        return from_vec<T>(vmin_u32(to_vec(lhs), to_vec(rhs)));
                    else if constexpr (N == 4)
                        return from_vec<T>(vminq_u32(to_vec(lhs), to_vec(rhs)));
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

    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    /**
     * @return number-maximum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto maxnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(
                        vmaxnm_f64(to_vec(lhs), to_vec(rhs))
                    );
                }
            #endif
            return {
                .val = internal::maxnm(lhs.val, rhs.val)
            };
        } else {

            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return from_vec<T>(vmaxnm_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4)
                    return from_vec<T>(vmaxnmq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2)
                    return from_vec<T>(vmaxnmq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vmaxnm_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vmaxnmq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(maxnm(cast<float>(lhs), cast<float>(rhs)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(maxnm(cast<float>(lhs), cast<float>(rhs)));
            }

            return join(
                maxnm(lhs.lo, rhs.lo),
                maxnm(lhs.hi, rhs.hi)
            );
        }
    }

    /**
     * @return number-minimum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto minnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(
                        vminnm_f64(to_vec(lhs), to_vec(rhs))
                    );
                }
            #endif
            return {
                .val = internal::minnm(lhs.val, rhs.val)
            };
        } else {

            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2)
                    return from_vec<T>(vminnm_f32(to_vec(lhs), to_vec(rhs)));
                else if constexpr (N == 4)
                    return from_vec<T>(vminnmq_f32(to_vec(lhs), to_vec(rhs)));
            } else if constexpr (std::same_as<T, double>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 2)
                    return from_vec<T>(vminnmq_f64(to_vec(lhs), to_vec(rhs)));
                #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vminnm_f16(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vminnmq_f16(to_vec(lhs), to_vec(rhs)));
                }
                #else
                return cast<T>(minnm(cast<float>(lhs), cast<float>(rhs)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(minnm(cast<float>(lhs), cast<float>(rhs)));
            }

            return join(
                minnm(lhs.lo, rhs.lo),
                minnm(lhs.hi, rhs.hi)
            );
        }
    }

// MARK: Pairwise Maximum
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmax(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return from_vec<T>(
                    vpmax_f32(to_vec(x), to_vec(y))
                ); 
            } else if constexpr (std::same_as<T, double>) {
                return from_vec<T>(
                    vpmaxq_f64(to_vec(x), to_vec(y))
                ); 
            }
            #endif
            return { { .val = std::max(x.lo.val, x.hi.val) }, { .val = std::max(y.lo.val, y.hi.val) } };
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 4) {
                    return from_vec<T>(
                        vpmaxq_f32(to_vec(x), to_vec(y))
                    ); 
                }
                #endif
            } else if constexpr (std::same_as<T, double>) {
                // do nothing
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vpmax_f16(to_vec(x), to_vec(y)));
                #ifdef UI_CPU_ARM64
                } else if constexpr (N == 8) {
                    return from_vec<T>(vpmaxq_f16(to_vec(x), to_vec(y)));
                #endif
                }
                #else
                return cast<T>(pmax(cast<float>(x), cast<float>(y)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(pmax(cast<float>(x), cast<float>(y)));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vpmax_s8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vpmaxq_s8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vpmax_s16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vpmaxq_s16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vpmax_s32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vpmaxq_s32(to_vec(x), to_vec(y))
                        ); 
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vpmax_u8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vpmaxq_u8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vpmax_u16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vpmaxq_u16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vpmax_u32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vpmaxq_u32(to_vec(x), to_vec(y))
                        ); 
                    }
                } 

            }   

            return join(
                pmax(x.lo, x.hi),
                pmax(y.lo, y.hi)
            );
        }
    }

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmax_t op
    ) noexcept -> T {
        if constexpr (N == 1) {
            return v.val;
        } else if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpmaxs_f32(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpmaxqd_f64(to_vec(v)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return T(fold(cast<float>(v)));
                }
            #endif
            return static_cast<T>(std::max(v.lo.val, v.hi.val));
        } else {
            return std::max(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    }

    /**
     * @return pairwise number-maximum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmaxnm(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(vpmaxnm_f32(to_vec(x), to_vec(y)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(vpmaxnmq_f64(to_vec(x), to_vec(y)));
                }
            #endif
            return {
                {.val = internal::maxnm(x.lo.val, x.hi.val)},
                {.val = internal::maxnm(y.lo.val, y.hi.val)}
            };
        } else {
            #ifdef UI_ALWAYS_INLINE
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vpmaxnmq_f32(to_vec(x), to_vec(y)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                    if constexpr (N == 4) {
                        return from_vec<T>(vpmaxnm_f16(to_vec(x), to_vec(y)));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vpmaxnmq_f16(to_vec(x), to_vec(y)));
                    }
                #else
                return cast<T>(pmaxnm(cast<float>(x), cast<float>(y)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(pmaxnm(cast<float>(x), cast<float>(y)));
            }
            #endif
            return join(
                pmaxnm(x.lo, x.hi),
                pmaxnm(y.lo, y.hi)
            );
        }
    }

    /**
     * @return pairwise number-maximum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmaxnm_t op
    ) noexcept -> T {
        if constexpr (N == 1) {
            return v.val;
        } else if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpmaxnms_f32(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpmaxnmqd_f64(to_vec(v)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return T(fold(cast<float>(v)));
                }
            #endif
            return static_cast<T>(internal::maxnm(v.lo.val, v.hi.val));
        } else {
            return internal::maxnm(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    }
// !MARK

// MARK: Pairwise Minimum
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmin(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                return from_vec<T>(
                    vpmin_f32(to_vec(x), to_vec(y))
                ); 
            } else if constexpr (std::same_as<T, double>) {
                return from_vec<T>(
                    vpminq_f64(to_vec(x), to_vec(y))
                ); 
            }
            #endif
            return { { .val = std::min(x.lo.val, x.hi.val) }, { .val = std::min(y.lo.val, y.hi.val) } };
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 4) {
                    return from_vec<T>(
                        vpminq_f32(to_vec(x), to_vec(y))
                    ); 
                }
                #endif
            } else if constexpr (std::same_as<T, double>) {
                // do nothing
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vpmin_f16(to_vec(x), to_vec(y)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vpminq_f16(to_vec(x), to_vec(y)));
                }
                #else
                return cast<T>(pmin(cast<float>(x), cast<float>(y)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(pmin(cast<float>(x), cast<float>(y)));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vpmin_s8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vpminq_s8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vpmin_s16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vpminq_s16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vpmin_s32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vpminq_s32(to_vec(x), to_vec(y))
                        ); 
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec<T>(
                            vpmin_u8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec<T>(
                            vpminq_u8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(
                            vpmin_u16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec<T>(
                            vpminq_u16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec<T>(
                            vpmin_u32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec<T>(
                            vpminq_u32(to_vec(x), to_vec(y))
                        ); 
                    }
                } 

            }   

            return join(
                pmin(x.lo, y.lo),
                pmin(x.hi, y.hi)
            );
        }
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmin_t op
    ) noexcept -> T {
        if constexpr (N == 1) {
            return v.val;
        } else if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpmins_f32(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpminqd_f64(to_vec(v)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return T(fold(cast<float>(v)));
                }
            #endif
            return static_cast<T>(std::min(v.lo.val, v.hi.val));
        } else {
            return std::min(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    }

    /**
     * @return pairwise number-minimum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pminnm(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(vpminnm_f32(to_vec(x), to_vec(y)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(vpminnmq_f64(to_vec(x), to_vec(y)));
                }
            #endif
            return {
                {.val = internal::minnm(x.lo.val, x.hi.val)},
                {.val = internal::minnm(y.lo.val, y.hi.val)}
            };
        } else {
            #ifdef UI_ALWAYS_INLINE
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 4) {
                    return from_vec<T>(vpminnmq_f32(to_vec(x), to_vec(y)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vpminnm_f16(to_vec(x), to_vec(y)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vpminnmq_f16(to_vec(x), to_vec(y)));
                }
                #else
                return cast<T>(pminnm(cast<float>(x), cast<float>(y)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return cast<T>(pminnm(cast<float>(x), cast<float>(y)));
            }
            #endif
            return join(
                pminnm(x.lo, y.lo),
                pminnm(x.hi, y.hi)
            );
        }
    }

    /**
     * @return pairwise number-minimum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pminnm_t op
    ) noexcept -> T {
        if constexpr (N == 1) {
            return v.val;
        } else if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpminnms_f32(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpminnmqd_f64(to_vec(v)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return T(fold(cast<float>(v)));
                }
            #endif
            return static_cast<T>(internal::minnm(v.lo.val, v.hi.val));
        } else {
            return internal::minnm(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    }

// !MARK

// MARK: Maximum across vector
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::max_t op
    ) noexcept -> T {
        using result_t = T;

        if constexpr (N == 1) {
            return v.val;
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vmaxv_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return static_cast<result_t>(vmaxvq_f32(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vmaxvq_f64(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return std::bit_cast<result_t>(vmaxv_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return std::bit_cast<result_t>(vmaxvq_f16(to_vec(v)));
                }
                #else
                return T(fold(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return T(fold(cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<result_t>(vmaxv_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return static_cast<result_t>(vmaxvq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<result_t>(vmaxv_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return static_cast<result_t>(vmaxvq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<result_t>(vmaxv_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return static_cast<result_t>(vmaxvq_s32(to_vec(v)));
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<result_t>(vmaxv_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return static_cast<result_t>(vmaxvq_u8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<result_t>(vmaxv_u16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return static_cast<result_t>(vmaxvq_u16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<result_t>(vmaxv_u32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return static_cast<result_t>(vmaxvq_u32(to_vec(v)));
                    }
                }
            }
            #endif
            return std::max(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    } 
    
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::maxnm_t op
    ) noexcept -> T {
        using result_t = T;

        if constexpr (N == 1) {
            return v.val;
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vmaxnmv_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return static_cast<result_t>(vmaxnmvq_f32(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vmaxnmvq_f64(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return std::bit_cast<result_t>(vmaxnmv_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return std::bit_cast<result_t>(vmaxnmvq_f16(to_vec(v)));
                }
                #else
                return T(fold(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return T(fold(cast<float>(v), op));
            }
            #endif
            return internal::maxnm(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    }

// !MARK

// MARK: Minimum across vector
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::min_t op
    ) noexcept -> T {
        using result_t = T;

        if constexpr (N == 1) {
            return v.val;
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vminv_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return static_cast<result_t>(vminvq_f32(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vminvq_f64(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return std::bit_cast<result_t>(vminv_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return std::bit_cast<result_t>(vminvq_f16(to_vec(v)));
                }
                #else
                return T(fold(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return T(fold(cast<float>(v), op));
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<result_t>(vminv_s8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return static_cast<result_t>(vminvq_s8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<result_t>(vminv_s16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return static_cast<result_t>(vminvq_s16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<result_t>(vminv_s32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return static_cast<result_t>(vminvq_s32(to_vec(v)));
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return static_cast<result_t>(vminv_u8(to_vec(v)));
                    } else if constexpr (N == 16) {
                        return static_cast<result_t>(vminvq_u8(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return static_cast<result_t>(vminv_u16(to_vec(v)));
                    } else if constexpr (N == 8) {
                        return static_cast<result_t>(vminvq_u16(to_vec(v)));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return static_cast<result_t>(vminv_u32(to_vec(v)));
                    } else if constexpr (N == 4) {
                        return static_cast<result_t>(vminvq_u32(to_vec(v)));
                    }
                }
            }
            #endif
            return std::min(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    } 
    
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::minnm_t op
    ) noexcept -> T {
        using result_t = T;

        if constexpr (N == 1) {
            return v.val;
        } else {
            #ifdef UI_CPU_ARM64
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vminnmv_f32(to_vec(v)));
                } else if constexpr (N == 4) {
                    return static_cast<result_t>(vminnmvq_f32(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return static_cast<result_t>(vminnmvq_f64(to_vec(v)));
                }
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return std::bit_cast<result_t>(vminnmv_f16(to_vec(v)));
                } else if constexpr (N == 8) {
                    return std::bit_cast<result_t>(vminnmvq_f16(to_vec(v)));
                }
                #else
                return T(fold(cast<float>(v), op));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                return T(fold(cast<float>(v), op));
            }
            #endif
            return internal::minnm(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    }
// !MARK

} // namespace ui::arm::neon

#endif // AMT_UI_ARCH_ARM_MINMAX_HPP

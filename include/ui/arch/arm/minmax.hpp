#ifndef AMT_UI_ARCH_ARM_MINMAX_HPP
#define AMT_UI_ARCH_ARM_MINMAX_HPP

#include "cast.hpp"
#include "ui/base.hpp"
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
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;
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
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;
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

    namespace internal {
        template <std::floating_point T>
        UI_ALWAYS_INLINE static constexpr auto maxnm(T a, T b) noexcept -> T {
            if (std::isnan(a)) return { .val = b };
            if (std::isnan(b)) return { .val = a };
            return std::max(a, b);
        }

        template <std::floating_point T>
        UI_ALWAYS_INLINE static constexpr auto minnm(T a, T b) noexcept -> T {
            if (std::isnan(a)) return { .val = b };
            if (std::isnan(b)) return { .val = a };
            return std::min(a, b);
        }
    } // namespace internal

    /**
     * @return number-maximum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto maxnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return std::bit_cast<ret_t>(
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

    /**
     * @return number-minimum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto minnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        using ret_t = Vec<N, T>;

        if constexpr (N == 1) {
            #ifdef UI_CPU_ARM64
                if constexpr (std::same_as<T, double>) {
                    return std::bit_cast<ret_t>(
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
                return from_vec(
                    vpmax_f32(to_vec(x), to_vec(y))
                ); 
            } else if constexpr (std::same_as<T, double>) {
                return from_vec(
                    vpmaxq_f64(to_vec(x), to_vec(y))
                ); 
            }
            #endif
            return { { .val = std::max(x.lo.val, x.hi.val) }, { .val = std::max(y.lo.val, y.hi.val) } };
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 4) {
                    return from_vec(
                        vpmaxq_f32(to_vec(x), to_vec(y))
                    ); 
                }
                #endif
            } else if constexpr (std::same_as<T, double>) {
                // do nothing
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(
                            vpmax_s8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec(
                            vpmaxq_s8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpmax_s16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpmaxq_s16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpmax_s32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec(
                            vpmaxq_s32(to_vec(x), to_vec(y))
                        ); 
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(
                            vpmax_u8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec(
                            vpmaxq_u8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpmax_u16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpmaxq_u16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpmax_u32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec(
                            vpmaxq_u32(to_vec(x), to_vec(y))
                        ); 
                    }
                } 

            }   

            return join(
                pmax(x.lo, y.lo),
                pmax(x.hi, y.hi)
            );
        }
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmax(
        Vec<N, T> const& v
    ) noexcept -> T {
        if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpmaxs_f32(from_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpmaxqd_f64(from_vec(v)));
                }
            #endif
            return static_cast<T>(std::max(v.lo.val, v.hi.val));
        } else {
            return join(
                pmax(v.lo),
                pmax(v.hi)
            );
        }
    }

    /**
     * @return pairwise number-minimum avoiding "NaN"
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
                    return from_vec(vpmaxnm_f32(to_vec(x), to_vec(y)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec(vpmaxnmq_f64(to_vec(x), to_vec(y)));
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
                    return from_vec(vpmaxnmq_f32(to_vec(x), to_vec(y)));
                }
            }
            #endif
            return join(
                pmaxnm(x.lo, y.lo),
                pmaxnm(x.hi, y.hi)
            );
        }
    }

    /**
     * @return pairwise number-minimum avoiding "NaN"
    */
    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmaxnm(
        Vec<N, T> const& v
    ) noexcept -> T {
        if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpmaxnms_f32(from_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpmaxnmqd_f64(from_vec(v)));
                }
            #endif
            return static_cast<T>(std::max(v.lo.val, v.hi.val));
        } else {
            return join(
                pmaxnm(v.lo),
                pmaxnm(v.hi)
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
                return from_vec(
                    vpmin_f32(to_vec(x), to_vec(y))
                ); 
            } else if constexpr (std::same_as<T, double>) {
                return from_vec(
                    vpminq_f64(to_vec(x), to_vec(y))
                ); 
            }
            #endif
            return { { .val = std::min(x.lo.val, x.hi.val) }, { .val = std::min(y.lo.val, y.hi.val) } };
        } else {
            if constexpr (std::same_as<T, float>) {
                #ifdef UI_CPU_ARM64
                if constexpr (N == 4) {
                    return from_vec(
                        vpminq_f32(to_vec(x), to_vec(y))
                    ); 
                }
                #endif
            } else if constexpr (std::same_as<T, double>) {
                // do nothing
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(
                            vpmin_s8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec(
                            vpminq_s8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpmin_s16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpminq_s16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpmin_s32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec(
                            vpminq_s32(to_vec(x), to_vec(y))
                        ); 
                    }
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(
                            vpmin_u8(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 16) {
                        return from_vec(
                            vpminq_u8(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(
                            vpmin_u16(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 8) {
                        return from_vec(
                            vpminq_u16(to_vec(x), to_vec(y))
                        ); 
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(
                            vpmin_u32(to_vec(x), to_vec(y))
                        ); 
                    } else if constexpr (N == 4) {
                        return from_vec(
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

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmin(
        Vec<N, T> const& v
    ) noexcept -> T {
        if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpmins_f32(from_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpminqd_f64(from_vec(v)));
                }
            #endif
            return static_cast<T>(std::min(v.lo.val, v.hi.val));
        } else {
            return join(
                pmin(v.lo),
                pmin(v.hi)
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
                    return from_vec(vpminnm_f32(to_vec(x), to_vec(y)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec(vpminnmq_f64(to_vec(x), to_vec(y)));
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
                    return from_vec(vpminnmq_f32(to_vec(x), to_vec(y)));
                }
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
    UI_ALWAYS_INLINE auto pminnm(
        Vec<N, T> const& v
    ) noexcept -> T {
        if constexpr (N == 2) {
            #ifdef UI_ALWAYS_INLINE
                if constexpr (std::same_as<T, float>) {
                    return static_cast<T>(vpminnms_f32(from_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return static_cast<T>(vpminnmqd_f64(from_vec(v)));
                }
            #endif
            return static_cast<T>(std::min(v.lo.val, v.hi.val));
        } else {
            return join(
                pminnm(v.lo),
                pminnm(v.hi)
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
    
    template <std::size_t N, typename T>
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
    
    template <std::size_t N, typename T>
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
            }
            #endif
            return internal::minnm(
                fold(v.lo, op),
                fold(v.hi, op)
            );
        }
    }
// !MARK

} // namespace ui::arm

#endif // AMT_UI_ARCH_ARM_MINMAX_HPP

#ifndef AMT_UI_ARCH_ARM_LOAD_HPP
#define AMT_UI_ARCH_ARM_LOAD_HPP

#include "cast.hpp"
#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace ui::arm::neon { 
    
    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto load(T val) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(val) };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (N == 2) {
                    return from_vec(vdup_n_f32(val));
                } else if constexpr (N == 4) {
                    return from_vec(vdupq_n_f32(val));
                }
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (N == 2) {
                    return from_vec(vdupq_n_f64(val));
                }
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vdup_n_f16(std::bit_cast<float16_t>(val)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vdupq_n_f16(std::bit_cast<float16_t>(val)));
                }
                #else
                return std::bit_cast<Vec<N, T>>(load<N>(std::bit_cast<std::uint16_t>(val.data)));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                #ifdef UI_HAS_BFLOAT_16
                if constexpr (N == 4) {
                    return from_vec<T>(vdup_n_bf16(std::bit_cast<bfloat16_t>(val)));
                } else if constexpr (N == 8) {
                    return from_vec<T>(vdupq_n_bf16(std::bit_cast<bfloat16_t>(val)));
                }
                #else
                return std::bit_cast<Vec<N, T>>(load<N>(std::bit_cast<std::uint16_t>(val.data)));
                #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(vdup_n_s8(val));
                    } else if constexpr (N == 16) {
                        return from_vec(vdupq_n_s8(val));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(vdup_n_s16(val));
                    } else if constexpr (N == 8) {
                        return from_vec(vdupq_n_s16(val));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_n_s32(val));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_n_s32(val));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec(vdupq_n_s64(val));
                    }
                }
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (N == 8) {
                        return from_vec(vdup_n_u8(val));
                    } else if constexpr (N == 16) {
                        return from_vec(vdupq_n_u8(val));
                    }
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (N == 4) {
                        return from_vec(vdup_n_u16(val));
                    } else if constexpr (N == 8) {
                        return from_vec(vdupq_n_u16(val));
                    }
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_n_u32(val));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_n_u32(val));
                    }
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (N == 2) {
                        return from_vec(vdupq_n_u64(val));
                    }
                }
            }
            return join(
                load<N / 2>(val),
                load<N / 2>(val)
            );
        }
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
        requires (Lane < M)
    UI_ALWAYS_INLINE auto load(
        Vec<M, T> const& v
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return { .val = static_cast<T>(v[Lane]) };
        } else {
            if constexpr (std::same_as<T, float>) {
                if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_lane_f32(to_vec(v), Lane));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_lane_f32(to_vec(v), Lane));
                    }
                } else if constexpr (M == 4) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_laneq_f32(to_vec(v), Lane));
                    } else if constexpr (N == 4) {
                        return from_vec(vdupq_laneq_f32(to_vec(v), Lane));
                    }
                } else if constexpr (M > 4) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                } 
            #ifdef UI_CPU_ARM64
            } else if constexpr (std::same_as<T, double>) {
                if constexpr (M == 1) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_lane_f64(to_vec(v), Lane));
                    }
                } else if constexpr (M == 2) {
                    if constexpr (N == 2) {
                        return from_vec(vdup_laneq_f64(to_vec(v), Lane));
                    }
                } else if constexpr (M > 2) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                } 
            #endif
            } else if constexpr (std::same_as<T, float16>) {
                #ifdef UI_HAS_FLOAT_16
                if constexpr (M == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_lane_f16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_lane_f16(to_vec(v), Lane));
                    }
                } else if constexpr (M == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_laneq_f16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_laneq_f16(to_vec(v), Lane));
                    }
                } else if constexpr (M > 4) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                }
                #else
                auto temp = std::bit_cast<Vec<N, std::uint16_t>>(v);
                return std::bit_cast<Vec<N, T>>(load<N, Lane>(temp));
                #endif
            } else if constexpr (std::same_as<T, bfloat16>) {
                #ifdef UI_HAS_BFLOAT_16
                if constexpr (M == 2) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_lane_bf16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_lane_bf16(to_vec(v), Lane));
                    }
                } else if constexpr (M == 4) {
                    if constexpr (N == 4) {
                        return from_vec<T>(vdup_laneq_bf16(to_vec(v), Lane));
                    } else if constexpr (N == 8) {
                        return from_vec<T>(vdupq_laneq_bf16(to_vec(v), Lane));
                    }
                } else if constexpr (M > 4) {
                    if constexpr (Lane < M / 2) {
                        return load<N, Lane>(v.lo);
                    } else {
                        return load<N, Lane - M / 2>(v.hi);
                    }
                }
                #else
                auto temp = std::bit_cast<Vec<N, std::uint16_t>>(v);
                return std::bit_cast<Vec<N, T>>(load<N, Lane>(temp));
                #endif
            } else if constexpr (std::is_signed_v<T>) {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (M == 8) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_lane_s8(to_vec(v), Lane));        
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_lane_s8(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 16) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_laneq_s8(to_vec(v), Lane));        
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_laneq_s8(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 16) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_lane_s16(to_vec(v), Lane));        
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_lane_s16(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_laneq_s16(to_vec(v), Lane));        
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_laneq_s16(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 8) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_lane_s32(to_vec(v), Lane));        
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_lane_s32(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_laneq_s32(to_vec(v), Lane));        
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_laneq_s32(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (M == 1) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_lane_s64(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_laneq_s64(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                #endif
                } 
            } else {
                if constexpr (sizeof(T) == 1) {
                    if constexpr (M == 8) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_lane_u8(to_vec(v), Lane));        
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_lane_u8(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 16) {
                        if constexpr (N == 8) {
                            return from_vec(vdup_laneq_u8(to_vec(v), Lane));        
                        } else if constexpr (N == 16) {
                            return from_vec(vdupq_laneq_u8(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 16) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 2) {
                    if constexpr (M == 4) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_lane_u16(to_vec(v), Lane));        
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_lane_u16(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 8) {
                        if constexpr (N == 4) {
                            return from_vec(vdup_laneq_u16(to_vec(v), Lane));        
                        } else if constexpr (N == 8) {
                            return from_vec(vdupq_laneq_u16(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 8) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                } else if constexpr (sizeof(T) == 4) {
                    if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_lane_u32(to_vec(v), Lane));        
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_lane_u32(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 4) {
                        if constexpr (N == 2) {
                            return from_vec(vdup_laneq_u32(to_vec(v), Lane));        
                        } else if constexpr (N == 4) {
                            return from_vec(vdupq_laneq_u32(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    }
                #ifdef UI_CPU_ARM64
                } else if constexpr (sizeof(T) == 8) {
                    if constexpr (M == 1) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_lane_u64(to_vec(v), Lane));        
                        }
                    } else if constexpr (M == 2) {
                        if constexpr (N == 2) {
                            return from_vec(vdupq_laneq_u64(to_vec(v), Lane));        
                        }
                    } else if constexpr (M > 4) {
                        if constexpr (Lane < M / 2) {
                            return load<N, Lane>(v.lo);
                        } else {
                            return load<N, Lane - M / 2>(v.hi);
                        }
                    } 
                #endif
                }
            }
            return join(
                load<N / 2, Lane>(v),
                load<N / 2, Lane>(v)
            );
        }
    }

} // namespace ui::arm::neon;

#endif // AMT_UI_ARCH_ARM_LOAD_HPP

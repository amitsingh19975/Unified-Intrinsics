#ifndef AMT_UI_ARCH_WASM_MINMAX_HPP
#define AMT_UI_ARCH_WASM_MINMAX_HPP

#include "cast.hpp"
#include "../emul/minmax.hpp"
#include "bit.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto max(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N <= 2) {
            return emul::max(lhs, rhs);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_max(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_max(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(max(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_max(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_max(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_i32x4_max(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_u8x16_max(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_u16x8_max(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_u32x4_max(to_vec(lhs), to_vec(rhs)));
                    }
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return max(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                max<false>(lhs.lo, rhs.lo),
                max<false>(lhs.hi, rhs.hi)
            );
        }
    }

    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto min(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        if constexpr (N <= 2) {
            return emul::min(lhs, rhs);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_min(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_min(to_vec(lhs), to_vec(rhs)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(min(cast<float>(lhs), cast<float>(rhs)));
                } else if constexpr (std::is_signed_v<T>) {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_min(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_min(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_i32x4_min(to_vec(lhs), to_vec(rhs)));
                    }
                } else {
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_u8x16_min(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_u16x8_min(to_vec(lhs), to_vec(rhs)));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_u32x4_min(to_vec(lhs), to_vec(rhs)));
                    }
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return min(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                min<false>(lhs.lo, rhs.lo),
                min<false>(lhs.hi, rhs.hi)
            );
        }
    }

    /**
     * @return number-maximum avoiding "NaN"
    */
    template <bool Merge = true, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto maxnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        using mtype = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::maxnm(lhs, rhs);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = wasm_f32x4_max(l, r);
                    auto mask_l = wasm_f32x4_ne(l, l); 
                    auto mask_r = wasm_f32x4_ne(r, r); 
                    auto fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_r),
                        from_vec<T>(l),
                        from_vec<T>(mx)
                    ); // if nan then l otherwise mx
                    fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_l),
                        from_vec<T>(r),
                        fixed_mx
                    ); // if nan then r otherwise fixed_mx
                    return fixed_mx;
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = wasm_f64x2_max(l, r);
                    auto mask_l = wasm_f64x2_ne(l, l); 
                    auto mask_r = wasm_f64x2_ne(r, r); 
                    auto fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_r),
                        from_vec<T>(l),
                        from_vec<T>(mx)
                    ); // if nan then l otherwise mx
                    fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_l),
                        from_vec<T>(r),
                        fixed_mx
                    ); // if nan then r otherwise fixed_mx
                    return fixed_mx;
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(maxnm(cast<float>(lhs), cast<float>(rhs)));
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return maxnm(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                maxnm<false>(lhs.lo, rhs.lo),
                maxnm<false>(lhs.hi, rhs.hi)
            );
        }
    }

    /**
     * @return number-minimum avoiding "NaN"
    */
    template <bool Merge = true, std::size_t N, std::floating_point T>
    UI_ALWAYS_INLINE auto minnm(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(lhs);
        using mtype = mask_inner_t<T>;
        if constexpr (N == 1) {
            return emul::minnm(lhs, rhs);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                if constexpr (std::same_as<T, float>) {
                    auto mx = wasm_f32x4_min(l, r);
                    auto mask_l = wasm_f32x4_ne(l, l); 
                    auto mask_r = wasm_f32x4_ne(r, r); 
                    auto fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_r),
                        from_vec<T>(l),
                        from_vec<T>(mx)
                    ); // if nan then l otherwise mx
                    fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_l),
                        from_vec<T>(r),
                        fixed_mx
                    ); // if nan then r otherwise fixed_mx
                    return fixed_mx;
                } else if constexpr (std::same_as<T, double>) {
                    auto mx = wasm_f64x2_min(l, r);
                    auto mask_l = wasm_f64x2_ne(l, l); 
                    auto mask_r = wasm_f64x2_ne(r, r); 
                    auto fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_r),
                        from_vec<T>(l),
                        from_vec<T>(mx)
                    ); // if nan then l otherwise mx
                    fixed_mx = bitwise_select(
                        from_vec<mtype>(mask_l),
                        from_vec<T>(r),
                        fixed_mx
                    ); // if nan then r otherwise fixed_mx
                    return fixed_mx;
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    return cast<T>(minnm(cast<float>(lhs), cast<float>(rhs)));
                }
            } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                return minnm(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                minnm<false>(lhs.lo, rhs.lo),
                minnm<false>(lhs.hi, rhs.hi)
            );
        }
    }

// MARK: Pairwise Maximum
    namespace internal {
        template <bool Merge = true, std::size_t N, typename T, typename Op>
            requires (N > 1)
        UI_ALWAYS_INLINE auto pminmax_helper(
            Vec<N, T> const& lhs,
            Vec<N, T> const& rhs,
            Op op,
            auto&& fn
        ) noexcept -> Vec<N, T> {
            static constexpr auto size = sizeof(lhs);
            if constexpr (N <= 2) {
                if constexpr (std::same_as<Op, op::max_t>) {
                    return emul::pmax(lhs, rhs);
                } else if constexpr (std::same_as<Op, op::min_t>) {
                    return emul::pmin(lhs, rhs);
                } else if constexpr (std::same_as<Op, op::maxnm_t>) {
                    return emul::pmaxnm(lhs, rhs);
                } else if constexpr (std::same_as<Op, op::minnm_t>) {
                    return emul::pminnm(lhs, rhs);
                }
            } else {
                if constexpr (size == sizeof(v128_t)) {
                    auto l = to_vec(lhs);
                    auto r = to_vec(rhs);
                    if constexpr (std::same_as<T, float>) {
                        // pminmax_helper([a0, a1, a2, a3], [b0, b1, b2, b3], max)
                        // [max(a0, a1), max(a2, a3), max(b0, b1), max(b2, b3)]
                        auto t0 = wasm_i32x4_shuffle(l, r, 0, 2, 4, 6);
                        auto t1 = wasm_i32x4_shuffle(l, r, 1, 3, 5, 7);

                        auto mx = fn(t0, t1);
                        return from_vec<T>(mx);
                    } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                        return cast<T>(pminmax_helper(cast<float>(lhs), cast<float>(rhs), op, fn));
                    } else if constexpr (sizeof(T) == 1) {
                        alignas(16) static constexpr std::uint8_t odd_mask[16] = {
                            1, 1, 3, 3, 5, 5, 7, 7,
                            9, 9, 11, 11, 13, 13, 15, 15 
                        };
                        alignas(16) static constexpr std::uint8_t even_pos_mask[16] = {
                            0, 2, 4, 6, 8, 10, 12, 14,
                            16, 18, 20, 22, 24, 26, 28, 30
                        };
                        auto odd_mask_vec = *reinterpret_cast<v128_t const*>(odd_mask);
                        auto even_pos_mask_vec = *reinterpret_cast<v128_t const*>(even_pos_mask);

                        // Extract odd indices from `l`
                        auto t0 = wasm_i8x16_swizzle(l, odd_mask_vec);
                        auto lmx = fn(t0, l);
                        auto lo = wasm_i8x16_swizzle(lmx, even_pos_mask_vec);

                        // Extract odd indices from `r`
                        auto t1 = wasm_i8x16_swizzle(r, odd_mask_vec);
                        auto rmx = fn(t1, r);
                        auto hi = wasm_i8x16_swizzle(rmx, even_pos_mask_vec);

                        return from_vec<T>(wasm_i64x2_shuffle(lo, hi, 0, 2));
                    } else if constexpr (sizeof(T) == 2) {
                        // [a00, a01, a10, a11, a20, a21, a30, a31, ...]
                        alignas(16) static constexpr std::uint8_t odd_mask[16] = {
                             2,  3,  2,  3,  6,  7,  6,  7,
                            10, 11, 10, 11, 14, 15, 14, 15
                        };
                        alignas(16) static constexpr std::uint8_t even_pos_mask[16] = {
                             0,  1,  4,  5,  8,  9, 12, 13,
                            16, 17, 20, 21, 24, 26, 30, 31
                        };
                        auto odd_mask_vec = *reinterpret_cast<v128_t const*>(odd_mask);
                        auto even_pos_mask_vec = *reinterpret_cast<v128_t const*>(even_pos_mask);

                        // Extract odd indices from `l`
                        auto t0 = wasm_i8x16_swizzle(l, odd_mask_vec);
                        auto lmx = fn(t0, l);
                        auto lo = wasm_i8x16_swizzle(lmx, even_pos_mask_vec);

                        // Extract odd indices from `r`
                        auto t1 = wasm_i8x16_swizzle(r, odd_mask_vec);
                        auto rmx = fn(t1, r);
                        auto hi = wasm_i8x16_swizzle(rmx, even_pos_mask_vec);

                        return from_vec<T>(wasm_i64x2_shuffle(lo, hi, 0, 2));
                    } else if constexpr (sizeof(T) == 4) {
                        auto t0 = wasm_i32x4_shuffle(l, l, 1, 1, 3, 3);
                        auto t1 = wasm_i32x4_shuffle(r, r, 1, 1, 3, 3);
                        auto lo = fn(t0, l);
                        auto hi = fn(t1, r);
                        return from_vec<T>(wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6));
                    }
                } else if constexpr (size * 2 == sizeof(v128_t) && Merge) {
                    return pminmax_helper(
                        join(lhs, rhs),
                        Vec<2 * N, T>{},
                        fn
                    ).lo;
                }

                return join(
                    pminmax_helper<false>(lhs.lo, lhs.hi, op, fn),
                    pminmax_helper<false>(rhs.lo, rhs.hi, op, fn)
                );
            }
        }

        template <bool Merge = true, typename O, std::size_t N, typename T>
            requires (N > 1)
        UI_ALWAYS_INLINE auto fold_helper(
            Vec<N, T> const& v,
            O op
        ) noexcept -> T {
            // TODO: Vectorize this code
            return emul::fold(v, op);
        }
    } // namespace internal

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmax(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(lhs, rhs, op::max_t{}, [](auto l, auto r) {
            using type = std::conditional_t<
                std::same_as<T, float16> || std::same_as<T, bfloat16>,
                float,
                T
            >;
            if constexpr (sizeof(l) == sizeof(v128_t)) {
                if constexpr (std::same_as<type, float>) return wasm_f32x4_max(l, r);
                else if constexpr (std::same_as<type, double>) return wasm_f64x2_max(l, r);
                else if constexpr (std::is_signed_v<type>) {
                    if constexpr (sizeof(type) == 1) return wasm_i8x16_max(l, r);
                    else if constexpr (sizeof(type) == 2) return wasm_i16x8_max(l, r);
                    else if constexpr (sizeof(type) == 4) return wasm_i32x4_max(l, r);
                    else return wasm_i64x2_gt(l, r);
                } else {
                    if constexpr (sizeof(type) == 1) return wasm_u8x16_max(l, r);
                    else if constexpr (sizeof(type) == 2) return wasm_u16x8_max(l, r);
                    else if constexpr (sizeof(type) == 4) return wasm_u32x4_max(l, r);
                    else return wasm_i64x2_gt(l, r);
                }
            }
        });
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmax_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::max_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pmax_t{});
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmaxnm(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(x, y, op::maxnm_t{}, [](auto l, auto r) {
            using type = std::conditional_t<
                std::same_as<T, float16> || std::same_as<T, bfloat16>,
                float,
                T
            >;
            return to_vec(maxnm(from_vec<type>(l), from_vec<type>(r)));
        });
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmaxnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::maxnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pmaxnm_t{});
    }
// !MARK

// MARK: Pairwise Minimum
    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pmin(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(lhs, rhs, op::min_t{}, [](auto l, auto r) {
            using type = std::conditional_t<
                std::same_as<T, float16> || std::same_as<T, bfloat16>,
                float,
                T
            >;
            if constexpr (sizeof(l) == sizeof(v128_t)) {
                if constexpr (std::same_as<type, float>) return wasm_f32x4_min(l, r);
                else if constexpr (std::same_as<type, double>) return wasm_f64x2_min(l, r);
                else if constexpr (std::is_signed_v<type>) {
                    if constexpr (sizeof(type) == 1) return wasm_i8x16_min(l, r);
                    else if constexpr (sizeof(type) == 2) return wasm_i16x8_min(l, r);
                    else if constexpr (sizeof(type) == 4) return wasm_i32x4_min(l, r);
                    else return wasm_i64x2_gt(l, r);
                } else {
                    if constexpr (sizeof(type) == 1) return wasm_u8x16_min(l, r);
                    else if constexpr (sizeof(type) == 2) return wasm_u16x8_min(l, r);
                    else if constexpr (sizeof(type) == 4) return wasm_u32x4_min(l, r);
                    else return wasm_i64x2_gt(l, r);
                }
            }
        });
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pmin_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, typename T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::min_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pmin_t{});
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto pminnm(
        Vec<N, T> const& x,
        Vec<N, T> const& y
    ) noexcept -> Vec<N, T> {
        return internal::pminmax_helper(x, y, op::minnm_t{}, [](auto l, auto r) {
            using type = std::conditional_t<
                std::same_as<T, float16> || std::same_as<T, bfloat16>,
                float,
                T
            >;
            return to_vec(minnm(from_vec<type>(l), from_vec<type>(r)));
        });
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        op::pminnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op);
    }

    template <std::size_t N, std::floating_point T>
        requires (N > 1)
    UI_ALWAYS_INLINE auto fold(
        Vec<N, T> const& v,
        [[maybe_unused]] op::minnm_t op
    ) noexcept -> T {
        return internal::fold_helper(v, op::pminnm_t{});
    }
// !MARK
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_MINMAX_HPP

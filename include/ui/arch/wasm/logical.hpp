#ifndef AMT_UI_ARCH_WASM_LOGICAL_HPP
#define AMT_UI_ARCH_WASM_LOGICAL_HPP

#include "cast.hpp"
#include "../emul/logical.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

// MARK: Bitwise And
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_and(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_and(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                return from_vec<T>(wasm_v128_and(l, r));
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return bitwise_and(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                bitwise_and<false>(lhs.lo, rhs.lo),
                bitwise_and<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise XOR
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_xor(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_xor(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                return from_vec<T>(wasm_v128_xor(l, r));
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return bitwise_xor(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                bitwise_xor<false>(lhs.lo, rhs.lo),
                bitwise_xor<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Negation
    template <bool Merge = true, std::size_t N, typename T>
        requires (std::is_floating_point_v<T> || std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto negate(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::negate(v);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_neg(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_neg(to_vec(v)));
                } else if constexpr (std::same_as<T, float16> || std::same_as<T, bfloat16>) {
                    static constexpr std::uint16_t sign_mask = 0x8000;
                    auto mask = Vec<N, std::uint16_t>::load(sign_mask);
                    return rcast<T>(bitwise_xor(rcast<std::uint16_t>(v), mask));
                } else {
                    auto m = to_vec(v);
                    if constexpr (sizeof(T) == 1) {
                        return from_vec<T>(wasm_i8x16_neg(m));
                    } else if constexpr (sizeof(T) == 2) {
                        return from_vec<T>(wasm_i16x8_neg(m));
                    } else if constexpr (sizeof(T) == 4) {
                        return from_vec<T>(wasm_i32x4_neg(m));
                    } else if constexpr (sizeof(T) == 8) {
                        return from_vec<T>(wasm_i64x2_neg(m));
                    }
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return negate(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            return join(
                negate<false>(v.lo),
                negate<false>(v.hi)
            );
        }
    }

    template <bool Merge, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto sat_sub(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T>;

    template <std::size_t N, std::integral T>
        requires (std::is_arithmetic_v<T> && std::is_signed_v<T>)
    UI_ALWAYS_INLINE auto sat_negate(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::sat_negate(v);
        } else {
            return sat_sub<true>(Vec<N, T>{}, v);
        }
    }
// !MARK

// MARK: Bitwise Not
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_not(
        Vec<N, T> const& v 
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::bitwise_not(v);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto m = to_vec(v);
                return from_vec<T>(wasm_v128_not(m));
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return bitwise_not(
                    from_vec<T>(fit_to_vec(v))
                ).lo;
            }

            return join(
                bitwise_not<false>(v.lo),
                bitwise_not<false>(v.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise OR
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_or(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_or(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                return from_vec<T>(wasm_v128_or(l, r));
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return bitwise_or(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                bitwise_or<false>(lhs.lo, rhs.lo),
                bitwise_or<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise Not-And ~lhs & rhs
    template <bool Merge = true, std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_notand(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(lhs);
        if constexpr (N == 1) {
            return emul::bitwise_notand(lhs, rhs);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                auto l = to_vec(lhs);
                auto r = to_vec(rhs);
                return from_vec<T>(wasm_v128_andnot(r, l));
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return bitwise_notand(
                    from_vec<T>(fit_to_vec(lhs)),
                    from_vec<T>(fit_to_vec(rhs))
                ).lo;
            }

            return join(
                bitwise_notand<false>(lhs.lo, rhs.lo),
                bitwise_notand<false>(lhs.hi, rhs.hi)
            );
        }
    }
// !MARK

// MARK: Bitwise Or-Not lhs | ~rhs
    template <std::size_t N, std::integral T>
    UI_ALWAYS_INLINE auto bitwise_ornot(
        Vec<N, T> const& lhs,
        Vec<N, T> const& rhs
    ) noexcept -> Vec<N, T> {
        if constexpr (N == 1) {
            return emul::bitwise_ornot(lhs, rhs);
        } else {
            return bitwise_not(bitwise_notand(lhs, rhs));
        }
    }
// !MARK
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_LOGICAL_HPP

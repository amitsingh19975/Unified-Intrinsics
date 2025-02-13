#ifndef AMT_UI_ARCH_ARM_MATRIX_HPP
#define AMT_UI_ARCH_ARM_MATRIX_HPP

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <utility>

namespace ui {

    template <std::size_t R, std::size_t C, typename T>
        requires (R <= 4 && C <= 4)
    UI_ALWAYS_INLINE auto transpose(
        VecMat<R, C, T> const& m
    ) noexcept {
        static constexpr auto NR = maths::nearest_power_of_2(R);
        using res_t = VecMat<C, NR, T>;
        if constexpr (C == R) {
            if constexpr (R == 2 && C == R) {
                auto res = res_t{};
                auto c0 = transpose_low(m.val[0], m.val[1]);
                auto c1 = transpose_high(m.val[0], m.val[1]);
                res.val[0] = join(c0.lo, c1.lo); 
                res.val[1] = join(c0.hi, c1.hi); 
                return res;
            } else if constexpr (R == 4 && C == R) {
                auto res = res_t{};
                auto c0 = transpose_low(m.val[0], m.val[1]);
                auto c1 = transpose_high(m.val[0], m.val[1]);
                auto c2 = transpose_low(m.val[2], m.val[3]);
                auto c3 = transpose_high(m.val[2], m.val[3]);
                res.val[0] = join(c0.lo, c2.lo);
                res.val[1] = join(c1.lo, c3.lo);
                res.val[2] = join(c0.hi, c2.hi);
                res.val[3] = join(c1.hi, c3.hi);
                return res;
            }
        } else if constexpr (R == 2 && C == 4) {
            auto res = res_t{};
            auto c0 = transpose_low(m.val[0], m.val[1]);
            auto c1 = transpose_high(m.val[0], m.val[1]);
            res.val[0] = c0.lo;
            res.val[1] = c1.lo;
            res.val[2] = c0.hi;
            res.val[3] = c1.hi;
            return res;
        } else if constexpr (R == 4 && C == 2) {
            auto res = res_t{};
            auto c0 = transpose_low(m.val[0], m.val[1]);
            auto c1 = transpose_high(m.val[0], m.val[1]);
            auto c2 = transpose_low(m.val[2], m.val[3]);
            auto c3 = transpose_high(m.val[2], m.val[3]);
            res.val[0] = join(c0, c2);
            res.val[1] = join(c1, c3);
            return res;
        } else if constexpr (R == 1) {
            auto res = res_t{};
            for (auto i = 0ul; i < C; ++i) res(i,0) = m(0,i);
            return res;
        } else if constexpr (C == 1) {
            auto res = res_t{};
            for (auto i = 0ul; i < R; ++i) res(0, i) = m(i, 0);
            return res;
        } else if constexpr (R == 3) {
            auto temp = VecMat<4, C, T>::load(m.val[0], m.val[1], m.val[2]);
            return transpose(temp); 
        }
    }

    template <std::size_t N, typename T>
        requires (N > 4)
    UI_ALWAYS_INLINE auto transpose(
        VecMat<N, N, T> const& m
    ) noexcept {
        // | b0 | b1 |
        // -----------
        // | b2 | b3 |
        //     |
        //     v
        // | b0 | b2 |
        // -----------
        // | b1 | b3 |

        auto b01 = m.lo();
        auto b23 = m.hi();

        VecMat<N / 2, N / 2, T> b0, b1, b2, b3;

        for (auto i = 0ul; i < N / 2; ++i) {
            b0.val[i] = b01.val[i].lo;
            b1.val[i] = b01.val[i].hi;
            b2.val[i] = b23.val[i].lo;
            b3.val[i] = b23.val[i].hi;
        }

        auto t0 = transpose(b0);
        auto t1 = transpose(b1);
        auto t2 = transpose(b2);
        auto t3 = transpose(b3);
    
        auto t01 = join_cols(t0, t2);
        auto t23 = join_cols(t1, t3);

        return join_rows(t01, t23);
    }

    template <std::size_t R, std::size_t C, typename T>
        requires ((R > 4 || C > 4) && (R != C) && maths::is_power_of_2(R))
    UI_ALWAYS_INLINE auto transpose(
        VecMat<R, C, T> const& m
    ) noexcept {

        if constexpr (C <= 4) {
            // | b0 |
            // ------
            // | b2 |
            //     |
            //     v
            // | b0 | b2 |
            auto b0 = m.lo();
            auto b1 = m.hi();
            auto t0 = transpose(b0);
            auto t1 = transpose(b1);

            return join_cols(t0, t1);
        } else if constexpr (R <= 4) {
            // | b0 | b1 |
            //     |
            //     v
            // | b0 |
            // ------
            // | b1 |
            VecMat<R, C / 2, T> b0, b1;
            for (auto i = 0ul; i < R; ++i) {
                b0.val[i] = m.val[i].lo;
                b1.val[i] = m.val[i].hi;
            }
            auto t0 = transpose(b0);
            auto t1 = transpose(b1);

            return join_rows(t0, t1);
        } else {
            // | b0 | b1 |
            // -----------
            // | b2 | b3 |
            //     |
            //     v
            // | b0 | b2 |
            // -----------
            // | b1 | b3 |

            auto b01 = m.lo();
            auto b23 = m.hi();

            VecMat<R / 2, C / 2, T> b0, b1, b2, b3;

            for (auto i = 0ul; i < R / 2; ++i) {
                b0.val[i] = b01.val[i].lo;
                b1.val[i] = b01.val[i].hi;
                b2.val[i] = b23.val[i].lo;
                b3.val[i] = b23.val[i].hi;
            }

            auto t0 = transpose(b0);
            auto t1 = transpose(b1);
            auto t2 = transpose(b2);
            auto t3 = transpose(b3);

            auto t01 = join_cols(t0, t2);
            auto t23 = join_cols(t1, t3);

            return join_rows(t01, t23);
        }
    }
    
    /**
     * @brief   This implementation uses broadcast to multiply a matrix and does
     *          not transpose the matrix, So, the caller needs to ensure they're passing
     *          a transposed matrix as a second argument.
     * @param c Matrix that will be added to product
     * @param a Matrix with dims (N, K)
     * @param b Matrix with dims (K, M)
     * @return result of inner product or matrix multiplication with dims (N. M)
     */
    template <std::size_t N, std::size_t M, std::size_t K, typename T>
    UI_ALWAYS_INLINE auto mul(
        VecMat<N, M, T> const& c,
        VecMat<N, K, T> const& a,
        VecMat<K, M, T> const& b
    ) noexcept -> VecMat<N, M, T> {
        auto res = c;
        /*auto bt = transpose(b); // VecMat<M, K>*/
        // | a00 a01 a02 a03 |      | b00 b01 b02 b03 |
        // | a10 a11 a12 a13 |   x  | b10 b11 b12 b13 |
        // | a20 a21 a22 a23 |      | b20 b21 b22 b23 |
        // | a30 a31 a32 a33 |      | b30 b31 b32 b33 |
        // c00 = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30
        // c01 = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31
        // c02 = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32
        // c03 = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33

        constexpr auto iter = []<std::size_t... Ks>(
            std::index_sequence<Ks...>,
            Vec<K, T> const& a_,
            VecMat<K, M, T> const& b_
        ) {
            constexpr auto fmul = [](Vec<M, T> const& acc, Vec<M, T> const& a0, Vec<M, T> const& b0) {
                if constexpr (std::floating_point<T>) {
                    return fused_mul_acc(acc, a0, b0, op::add_t{});
                } else {
                    return mul_acc(acc, a0, b0, op::add_t{});
                }
            };
            auto c_ = Vec<M, T>{};
            ((c_ = fmul(c_, load<M, Ks>(a_), b_.val[Ks])),...);
            return c_;
        };
        for (auto i = 0ul; i < N; ++i) {
            auto a0 = a.val[i];
            res.val[i] = iter(std::make_index_sequence<K>{}, a0, b);
        }

        return res;
    }

    /**
     * @brief This implementation uses fold operation and transposes the matrix internally.
     * @param c Matrix that will be added to product
     * @param a Matrix with dims (N, K)
     * @param b Matrix with dims (K, M)
     * @return result of inner product or matrix multiplication with dims (N. M)
     */
    template <std::size_t N, std::size_t M, std::size_t K, typename T>
    UI_ALWAYS_INLINE auto mul2(
        VecMat<N, M, T> const& c,
        VecMat<N, K, T> const& a,
        VecMat<K, M, T> const& b
    ) noexcept -> VecMat<N, M, T> {
        auto res = c;
        auto bt = transpose(b); // VecMat<M, K>
        // | a00 a01 a02 a03 |      | b00 b01 b02 b03 |
        // | a10 a11 a12 a13 |   x  | b10 b11 b12 b13 |
        // | a20 a21 a22 a23 |      | b20 b21 b22 b23 |
        // | a30 a31 a32 a33 |      | b30 b31 b32 b33 |
        // c00 = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30
        // c01 = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31
        // c02 = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32
        // c03 = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33

        for (auto i = 0ul; i < N; ++i) {
            auto a0 = a.val[i];
            auto c0 = res.val[i];
            for (auto j = 0ul; j < M; ++j) {
                auto b0 = bt.val[j];
                c0[j] += pfold(fmul(a0, b0), op::add_t{});
            }
            res.val[i] = c0;
        }

        return res;
    }

    /**
     * @brief This implementation uses fold operation and transposes the matrix internally.
     * @param a Matrix with dims (N, K)
     * @param b Matrix with dims (K, M)
     * @return result of inner product or matrix multiplication with dims (N. M)
     */
    template <std::size_t N, std::size_t M, std::size_t K, typename T>
    UI_ALWAYS_INLINE auto mul2(
        VecMat<N, K, T> const& a,
        VecMat<K, M, T> const& b
    ) noexcept -> VecMat<N, M, T> {
        return mul2({}, a, b);
    }

    /**
     * @brief   This implementation uses broadcast to multiply a matrix and does
     *          not transpose the matrix, So, the caller needs to ensure they're passing
     *          a transposed matrix as a second argument.
     * @param a Matrix with dims (N, K)
     * @param b Matrix with dims (K, M)
     * @return result of inner product or matrix multiplication with dims (N. M)
     */
    template <std::size_t N, std::size_t M, std::size_t K, typename T>
    UI_ALWAYS_INLINE auto mul(
        VecMat<N, K, T> const& a,
        VecMat<K, M, T> const& b
    ) noexcept -> VecMat<N, M, T> {
        return mul({}, a, b);
    }
} // namespace ui

#endif // AMT_UI_ARCH_ARM_MATRIX_HPP

#ifndef AMT_UI_MODULAR_INV_HPP
#define AMT_UI_MODULAR_INV_HPP

#include "features.hpp"
#include "float.hpp"
#include "ui/maths.hpp"
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ui::maths {

    namespace internal {
        template <std::size_t Iter = 1, bool M = false, typename T>
            requires (std::is_arithmetic_v<T>)
        static constexpr auto calculate_reciprocal(T n, T nr = T(1)) noexcept -> T {
            constexpr auto iter = [](T n_, T nr_) {
                for (auto i = std::size_t{}; i < Iter; ++i) {
                    T t = 2 - n_ * nr_; 
                    if constexpr (M) {
                       t = t * nr_; 
                    } 
                    nr_ = t;
                }
                return nr_;
            };
            if constexpr (std::integral<T>) {
                return iter(n, nr);
            } else if constexpr (std::same_as<float16, T> || std::same_as<T, bfloat16>) {
                return T(calculate_reciprocal(float(n), float(nr)));
            } else {
                if (std::isnan(n)) return n;
                if (std::isinf(n)) return T{};
                if (std::isinf(nr)) return T(2);
                return iter(n, nr);
            }
        }

        template <std::size_t Iter = 1, std::floating_point T>
        static constexpr auto calculate_sqrt_inv(T n, T nr = T(1)) noexcept -> T {
            if (std::isnan(n)) return n;
            if (std::isinf(n) || std::isinf(nr)) return T{};
            for (auto i = std::size_t{}; i < Iter; ++i) {
                nr = nr * ((T(3) / T(2)) - ((n * nr * nr) / 2));
            }
            return nr;
        }

        template <std::size_t Iter = 1, std::integral T>
        static constexpr auto calculate_sqrt_inv(T n, T nr = T(1)) noexcept -> T {
            auto tn = std::uint64_t(n);
            auto tnr = std::uint64_t(nr);
            for (auto i = std::size_t{}; i < Iter; ++i) {
                tnr = tnr * ((3 - (tn * tnr * tnr)) >> 1);
            }
            return static_cast<T>(tnr);
        }
    } // namespace internal

    struct BinaryReciprocal {
    private:
        static constexpr auto lookup_table = []{
            std::array<std::uint8_t, 128> res;

            for (auto i = 0u; i < 128u; ++i) {
                auto inv = internal::calculate_reciprocal<3>(2 * i + 1); // only odd numbers
                res[i] = inv & 0xff;
            }

            return res;
        }();

    public:
        template <std::integral T>
        constexpr auto operator()(T n) const noexcept -> T {
            assert((n & 1) && "number should be an odd");

            auto val = static_cast<std::size_t>(n);
            // 1. initial guess
            // We divide by two since inverses are stored in previous position since we
            // calculate inverses for only odd numbers.
            std::size_t inv = lookup_table[(n & 0xff) >> 1];

            // 2. Refine approximation using Newton iterations
            //    Each iteration doubles the number of correct bits
            if constexpr (sizeof(T) == 2) {
                inv = (1 - val * inv) * inv + inv;  // 16 bits
            }
            if constexpr (sizeof(T) == 4) {
                inv = (1 - val * inv) * inv + inv;  // 32 bits
            }
            if constexpr (sizeof(T) == 8) {
                inv = (1 - val * inv) * inv + inv;  // 64 bits
            }

            return static_cast<T>(inv);
        }

        template <std::integral T>
        constexpr auto mod_inv(T n) const noexcept -> T {
            static constexpr auto max = std::numeric_limits<T>::max();
            if (n < 1) return max;
            if (n & 1) return this->operator()(n);

            auto n_z = static_cast<std::size_t>(std::countr_zero(n));
            constexpr auto mod_z = sizeof(T) * 8;
            assert((n_z <= mod_z) && "Mod should be bigger than the 'n'");

            if ((mod_z - n_z) < 1) return T(1);
            auto n_odd = n >> n_z;
            auto inv = static_cast<std::size_t>(this->operator()(n_odd));
            return static_cast<T>(inv << n_z);
        }

        template <std::size_t S = 1, std::integral T>
        constexpr auto iestimate(T n) const noexcept -> T {
            static constexpr auto max = std::numeric_limits<T>::max();
            if (n == 0) return max;
            #ifndef UI_HAS_INT128
            static_assert(sizeof(T) < 8, "64bit integers are not supported");
            #else
            if constexpr (sizeof(T) == 8) {
                auto temp = (ui::uint128_t(1) << (128 - S)) / n;
                return temp > max ? max : static_cast<T>(temp);
            }
            #endif

            if constexpr (sizeof(T) < 8) {
                auto temp = (std::uint64_t(1) << (64 - S)) / n;
                return temp > max ? max : static_cast<T>(temp);
            }
        }

        constexpr auto estimate(float16 n) const noexcept -> float16 {
            return estimate(float(n));
        }

        constexpr auto estimate(bfloat16 n) const noexcept -> bfloat16 {
            return estimate(float(n));
        }

        constexpr auto estimate(float n) const noexcept -> float {
            if (n == 0.f) return std::numeric_limits<float>::infinity();
            if (n == std::numeric_limits<float>::infinity()) return 0.0f;
            if (n == -std::numeric_limits<float>::infinity()) return 0.0f;
            if (n != n) return n;

            auto bits = std::bit_cast<std::uint32_t>(n);
            bits = ( 0xBE6EB3BEu - bits ) >> 1;
            auto temp = std::bit_cast<float>(bits);
            return temp * temp;
        }

        constexpr auto estimate(double n) const noexcept -> double {
            if (n == 0.0) return std::numeric_limits<double>::infinity();
            if (n == std::numeric_limits<double>::infinity()) return 0.0;
            if (n == -std::numeric_limits<double>::infinity()) return 0.0;
            if (n != n) return n;

            auto bits = std::bit_cast<std::uint64_t>(n);
            bits = ( 0XBFCDD6A18F6A6F52ul - bits ) >> 1;
            auto temp = std::bit_cast<double>(bits);
            return temp * temp;
        }

        template <std::integral T>
        constexpr auto isqrt_inv(T n) const noexcept -> T {
            auto res = maths::isqrt</*RoundUp=*/true>(n);
            // sqrt requres half the width. 2^((3 * W)/ - 1)/sqrt(n)
            static constexpr std::size_t S = (sizeof(T) * 8) / 2 + 1;
            auto e = iestimate<S>(res);
            return e;
        }

        constexpr auto sqrt_inv(float n) const noexcept -> float {
            if (n == 0) return std::numeric_limits<float>::infinity();
            auto bits = std::bit_cast<std::uint32_t>(n);
            auto temp = std::bit_cast<float>(0x5F1FFFF9 - (bits >> 1));
            return internal::calculate_sqrt_inv(n, temp);
        }

        constexpr auto sqrt_inv(float16 n) const noexcept -> float16 {
            auto temp = static_cast<float>(n);
            return sqrt_inv(temp);
        }

        constexpr auto sqrt_inv(bfloat16 n) const noexcept -> bfloat16 {
            auto temp = static_cast<float>(n);
            return sqrt_inv(temp);
        }

        constexpr auto sqrt_inv(double n) const noexcept -> double {
            if (n == 0) return std::numeric_limits<double>::infinity();
            auto bits = std::bit_cast<std::uint64_t>(n);
            auto temp = std::bit_cast<double>(0x5FE6EC85E7DE30DAull - (bits >> 1));
            return internal::calculate_sqrt_inv(n, temp);
        }
    };

} // namespace ui::maths

#endif // AMT_UI_MODULAR_INV_HPP

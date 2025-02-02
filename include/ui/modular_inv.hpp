#ifndef AMT_UI_MODULAR_INV_HPP
#define AMT_UI_MODULAR_INV_HPP

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
        template <std::size_t Iter = 1, typename T>
            requires (std::is_arithmetic_v<T>)
        static constexpr auto calculate_reciprocal(T n, T nr = T(1)) noexcept -> T {
            // INFO: Use Newton-Raphson iteration to find `nr` (modular multiplicative inverse)
            // total iterations required log2(number_of_bits) = log2(8) = 3	
            for (auto i = std::size_t{}; i < Iter; ++i) {
                nr *= 2 - n * nr; 
            }
            return nr;
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

            return static_cast<T>(inv & std::numeric_limits<T>::max());
        }

        template <std::integral T>
        constexpr auto estimate(T n) const noexcept -> T {
            assert(n != 0);
            if (n & 1) return this->operator()(n);

            auto n_z = static_cast<std::size_t>(std::countr_zero(n));
            constexpr auto mod_z = sizeof(T) * 8;
            assert((n_z <= mod_z) && "Mod should be bigger than the 'n'");

            auto n_odd = n >> n_z;
            auto inv = (mod_z - n_z) <= 1 ? std::size_t{1} : static_cast<std::size_t>(this->operator()(n_odd));
            return static_cast<T>(inv << n_z);
        }

        constexpr auto estimate(float n) const noexcept -> float {
            if (n == 0.f) return std::numeric_limits<float>::infinity();
            if (n == std::numeric_limits<float>::infinity()) return 0.0f;
            if (n == -std::numeric_limits<float>::infinity()) return 0.0f;
            if (n != n) return n;

            auto bits = std::bit_cast<std::uint32_t>(n);
            bits = ( 0xBE6EB3BEul - bits ) >> 1;
            auto temp = std::bit_cast<float>(bits);
            return temp * temp;
        }

        constexpr auto estimate(double n) const noexcept -> double {
            if (n == 0.0) return std::numeric_limits<double>::infinity();
            if (n == std::numeric_limits<double>::infinity()) return 0.0;
            if (n == -std::numeric_limits<double>::infinity()) return 0.0;
            if (n != n) return n;

            auto bits = std::bit_cast<std::uint64_t>(n);
            bits = ( 0XBFCDD6A18F6A6F52ull - bits ) >> 1;
            auto temp = std::bit_cast<double>(bits);
            return temp * temp;
        }

    };

} // namespace ui::maths

#endif // AMT_UI_MODULAR_INV_HPP

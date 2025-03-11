#ifndef AMT_UI_ARCH_WASM_SQRT_HPP
#define AMT_UI_ARCH_WASM_SQRT_HPP

#include "cast.hpp"
#include "../emul/sqrt.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    } // namespace internal

    template<bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto sqrt(
        Vec<N, T> const& v
    ) noexcept -> Vec<N, T> {
        static constexpr auto bits = sizeof(v);
        if constexpr (N == 1) {
            return emul::sqrt(v);
        } else {
            if constexpr (bits == sizeof(v128_t)) {
                if constexpr (std::same_as<T, float>) {
                    return from_vec<T>(wasm_f32x4_sqrt(to_vec(v)));
                } else if constexpr (std::same_as<T, double>) {
                    return from_vec<T>(wasm_f64x2_sqrt(to_vec(v)));
                } else if constexpr (::ui::internal::is_fp16<T>) {
                    return cast<T>(sqrt(cast<float>(v)));
                }
            } else if constexpr (bits * 2 == sizeof(v128_t) && Merge) {
                return sqrt(from_vec<T>(fit_to_vec(v))).lo;
            }

            return join(
                sqrt<false>(v.lo),
                sqrt<false>(v.hi)
            );
        }
    }
} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_SQRT_HPP

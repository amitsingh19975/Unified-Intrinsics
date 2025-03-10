#ifndef AMT_UI_ARCH_WASM_BIT_HPP
#define AMT_UI_ARCH_WASM_BIT_HPP

#include "cast.hpp"
#include "../emul/bit.hpp"
#include "logical.hpp"

namespace ui::wasm {
    namespace internal {
        using namespace ::ui::internal;
    }

// MARK: Bitwise select
    template <bool Merge = true, std::size_t N, typename T>
    UI_ALWAYS_INLINE auto bitwise_select(
        mask_t<N, T> const& cond,
        Vec<N, T> const& true_,
        Vec<N, T> const& false_
    ) noexcept -> Vec<N, T> {
        static constexpr auto size = sizeof(cond);
        if constexpr (N == 1) {
            return emul::bitwise_select(cond, true_, false_);
        } else {
            if constexpr (size == sizeof(v128_t)) {
                auto c = to_vec(cond);
                auto t = to_vec(true_);
                auto f = to_vec(false_);
                auto t0 = bitwise_and(cond, rcast<mask_inner_t<T>>(true_));
                auto t1 = bitwise_notand(cond, rcast<mask_inner_t<T>>(false_));
                return rcast<T>(bitwise_or(t0, t1));
            } else if constexpr (size * 2 == sizeof(v128_t)) {
                return bitwise_select(
                    from_vec<mask_inner_t<T>>(fit_to_vec(cond)),
                    from_vec<T>(fit_to_vec(true_)),
                    from_vec<T>(fit_to_vec(false_))
                ).lo;
            }
            return join(
                bitwise_select<false>(cond.lo, true_.lo, false_.lo),
                bitwise_select<false>(cond.hi, true_.hi, false_.hi)
            );
        }
    }
// !MARK

} // namespace ui::wasm

#endif // AMT_UI_ARCH_WASM_BIT_HPP

#ifndef AMT_ARCH_EMUL_INT_MASK_HPP
#define AMT_ARCH_EMUL_INT_MASK_HPP

#include "cast.hpp"
#include "shift.hpp"

namespace ui {
    template <std::size_t N, typename T>
    inline constexpr IntMask<N, T>::IntMask(mask_t<N, T> const& m) noexcept {
        using namespace emul;
        using mtype = mask_inner_t<T>;

        auto ext = rcast<mtype>(shift_right<7>(rcast<std::make_signed_t<T>>(m)));
        auto helper = [&ext]<std::size_t... Is>(std::index_sequence<Is...>) -> base_type {
            auto res = base_type{};
            ((res |= (base_type(ext[Is] & 1) << Is)),...);
            return res;
        };
        mask = helper(std::make_index_sequence<N>{});
    }
} // namespace ui
#endif // AMT_ARCH_EMUL_INT_MASK_HPP

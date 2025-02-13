#ifndef AMT_UI_ARCH_PERMUTE_HPP
#define AMT_UI_ARCH_PERMUTE_HPP

#include "cast.hpp"

namespace ui {

    template <std::size_t... Is, std::size_t N, typename T>
    UI_ALWAYS_INLINE static constexpr auto shuffle(
        Vec<N, T> const& x
    ) noexcept -> Vec<sizeof...(Is), T> {
        #ifndef UI_COMPILER_CLANG
            return to_vec<sizeof...(Is), T>(__builtin_shufflevector(to_vec(x), to_vec(x), Is...));
        #else
            return Vec<sizeof...(Is), T>::load(x[Is]...);
        #endif
    }
} // ui

#endif // AMT_UI_ARCH_PERMUTE_HPP 

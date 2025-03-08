#ifndef AMT_UI_ARCH_EMUL_PREFETCH_HPP
#define AMT_UI_ARCH_EMUL_PREFETCH_HPP

#include "../../base.hpp"
#include <cstdint>

namespace ui::emul {
    enum class PrefetchRW: std::int8_t {
        Read = 0,
        Write = 1,
        RW = 1
    };

    enum class PrefetchLocality: std::int8_t {
        None = 0,
        Low = 1,
        Medium = 2,
        High = 3
    };

    template <PrefetchRW RW = PrefetchRW::Read, PrefetchLocality Locality = PrefetchLocality::High, typename T>
    UI_ALWAYS_INLINE static constexpr auto prefetch(
        [[maybe_unused]] T const* const data
    ) noexcept {
        #if defined(UI_COMPILER_CLANG) || defined(UI_COMPILER_GCC)
            #if __has_builtin(__builtin_prefetch)
                __builtin_prefetch(data, static_cast<int>(RW), static_cast<int>(Locality));
            #endif
        #endif
    }

} // ui::emul

#endif // AMT_UI_ARCH_EMUL_PREFETCH_HPP 

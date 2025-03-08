#ifndef AMT_UI_ARCH_X86_PREFETCH_HPP
#define AMT_UI_ARCH_X86_PREFETCH_HPP

#include "../emul/prefetch.hpp"
#include <xmmintrin.h>

namespace ui::arm::neon {
    using emul::PrefetchRW, emul::PrefetchLocality;

    template <PrefetchRW RW = PrefetchRW::Read, PrefetchLocality Locality = PrefetchLocality::High, typename T>
    UI_ALWAYS_INLINE static constexpr auto prefetch(
        [[maybe_unused]] T const* const data
    ) noexcept {
    #if defined(UI_COMPILER_CLANG) || defined(UI_COMPILER_GCC)
        emul::prefetch<RW, Locality>(data);
    #else
        static constexpr auto selector = static_cast<int>(PrefetchLocality::High) - static_cast<int>(Locality);
        _mm_prefetch(data, selector);
    #endif
    }

} // ui::arm::neon

#endif // AMT_UI_ARCH_X86_PREFETCH_HPP 

#ifndef AMT_UI_ARCH_ARM_PREFETCH_HPP
#define AMT_UI_ARCH_ARM_PREFETCH_HPP

#include "../emul/prefetch.hpp"
#include <arm_neon.h>

namespace ui::arm::neon {
    using emul::PrefetchRW, emul::PrefetchLocality;

    template <PrefetchRW RW = PrefetchRW::Read, PrefetchLocality Locality = PrefetchLocality::High, typename T>
    UI_ALWAYS_INLINE static constexpr auto prefetch(
        [[maybe_unused]] T const* const data
    ) noexcept {
    #ifdef UI_COMPILER_MSVC 
        __prefetch(data);
    #elif defined(UI_COMPILER_CLANG) || defined(UI_COMPILER_GCC)
        emul::prefetch<RW, Locality>(data);
    #else
        asm volatile ("pld [%0]" : : "r" (data));
    #endif
    }

} // ui::arm::neon

#endif // AMT_UI_ARCH_ARM_PREFETCH_HPP 

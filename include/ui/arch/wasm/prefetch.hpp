#ifndef AMT_UI_ARCH_WASM_PREFETCH_HPP
#define AMT_UI_ARCH_WASM_PREFETCH_HPP

#include "../emul/prefetch.hpp"

namespace ui::wasm {
    using emul::PrefetchRW, emul::PrefetchLocality, emul::prefetch; 
} // ui::wasm

#endif // AMT_UI_ARCH_WASM_PREFETCH_HPP 

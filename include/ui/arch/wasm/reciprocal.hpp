#ifndef AMT_UI_ARCH_WASM_RECIPROCAL_HPP
#define AMT_UI_ARCH_WASM_RECIPROCAL_HPP

#include "../emul/reciprocal.hpp"

namespace ui::wasm {
    using emul::reciprocal_estimate,
        emul::reciprocal_refine,
        emul::sqrt_inv_estimate,
        emul::sqrt_inv_refine,
        emul::exponent_reciprocal_estimate;
} // ui::wasm

#endif // AMT_UI_ARCH_WASM_RECIPROCAL_HPP 

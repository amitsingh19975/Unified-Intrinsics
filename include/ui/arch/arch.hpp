#ifndef AMT_UI_ARCH_ARCH_HPP
#define AMT_UI_ARCH_ARCH_HPP

#include "../vec_headers.hpp"

namespace ui {
    enum class Arch {
        Unknown,
        Arm,
        x86,
        Wasm,
        Emul
    };
} // namespace ui

#if defined(UI_ARM_HAS_NEON) && !defined(UI_NO_NATIVE_VECTOR)
    #include "arm/abs.hpp"
    #include "arm/add.hpp"
    #include "arm/bit.hpp"
    #include "arm/cast.hpp"
    #include "arm/cmp.hpp"
    #include "arm/div.hpp"
    #include "arm/join.hpp"
    #include "arm/load.hpp"
    #include "arm/logical.hpp"
    #include "arm/manip.hpp"
    #include "arm/minmax.hpp"
    #include "arm/mul.hpp"
    #include "arm/reciprocal.hpp"
    #include "arm/rounding.hpp"
    #include "arm/shift.hpp"
    #include "arm/sub.hpp"
    #include "arm/sqrt.hpp"
    #include "arm/permute.hpp"
    #include "arm/prefetch.hpp"

    namespace ui {
        using namespace arm::neon;
        static constexpr auto ARCH_TYPE = Arch::Arm;
    }
    #define VEC_ARCH_NAME "ARM"

#elif (UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE41) && !defined(UI_NO_NATIVE_VECTOR)
    #include "x86/cast.hpp"
    #include "x86/load.hpp"
    #include "x86/cmp.hpp"
    #include "x86/logical.hpp"
    #include "x86/add.hpp"
    #include "x86/shift.hpp"
    #include "x86/sub.hpp"
    #include "x86/abs.hpp"
    #include "x86/minmax.hpp"
    #include "x86/mul.hpp"
    #include "x86/bit.hpp"
    #include "x86/rounding.hpp"
    #include "x86/div.hpp"
    #include "x86/sqrt.hpp"
    #include "x86/reciprocal.hpp"
    #include "x86/manip.hpp"
    #include "x86/permute.hpp"
    #include "x86/prefetch.hpp"

    namespace ui {
        using namespace x86;
        static constexpr auto ARCH_TYPE = Arch::x86;
    }
    #define VEC_ARCH_NAME "x86"
#elif defined(UI_EMPSCRIPTEN)
    #include "wasm/abs.hpp"
    #include "wasm/add.hpp"
    #include "wasm/bit.hpp"
    #include "wasm/cast.hpp"
    #include "wasm/cmp.hpp"
    #include "wasm/div.hpp"
    #include "wasm/load.hpp"
    #include "wasm/logical.hpp"
    #include "wasm/manip.hpp"
    #include "wasm/minmax.hpp"
    #include "wasm/mul.hpp"
    #include "emul/reciprocal.hpp"
    #include "wasm/rounding.hpp"
    #include "wasm/shift.hpp"
    #include "wasm/sub.hpp"
    #include "wasm/sqrt.hpp"
    #include "wasm/int_mask.hpp"
    #include "wasm/permute.hpp"
    #include "wasm/prefetch.hpp"
    namespace ui {
        using namespace wasm;
        static constexpr auto ARCH_TYPE = Arch::Wasm;
    }
    #define VEC_ARCH_NAME "Wasm"
#else
    #include "emul/abs.hpp"
    #include "emul/add.hpp"
    #include "emul/bit.hpp"
    #include "emul/cast.hpp"
    #include "emul/cmp.hpp"
    #include "emul/div.hpp"
    #include "emul/load.hpp"
    #include "emul/logical.hpp"
    #include "emul/manip.hpp"
    #include "emul/minmax.hpp"
    #include "emul/mul.hpp"
    #include "emul/reciprocal.hpp"
    #include "emul/rounding.hpp"
    #include "emul/shift.hpp"
    #include "emul/sub.hpp"
    #include "emul/sqrt.hpp"
    #include "emul/permute.hpp"
    #include "emul/int_mask.hpp"
    #include "emul/permute.hpp"
    #include "emul/prefetch.hpp"

    namespace ui {
        using namespace emul;
        static constexpr auto ARCH_TYPE = Arch::Emul;
    }
    #define VEC_ARCH_NAME "EMUL"
#endif

#include "cast_float.hpp"
#include "matrix.hpp"

#endif // AMT_UI_ARCH_ARCH_HPP


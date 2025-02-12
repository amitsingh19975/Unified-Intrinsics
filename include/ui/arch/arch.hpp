#ifndef AMT_UI_ARCH_ARCH_HPP
#define AMT_UI_ARCH_ARCH_HPP

#include "../vec_headers.hpp"

#if defined(UI_ARM_HAS_NEON)
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
namespace ui {
    using namespace arm::neon;
}
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

namespace ui {
    using namespace emul;
}


#endif
#include "cast_float.hpp"
#include "matrix.hpp"
#endif // AMT_UI_ARCH_ARCH_HPP

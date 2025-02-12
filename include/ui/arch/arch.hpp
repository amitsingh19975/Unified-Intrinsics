#ifndef AMT_UI_ARCH_ARCH_HPP
#define AMT_UI_ARCH_ARCH_HPP

#include "../features.hpp"

#if defined(UI_ARM_HAS_NEON)
#include "arm/abs.hpp"
#include "arm/add.hpp"
#include "arm/bit.hpp"
#include "arm/cast.hpp"
#include "arm/cast_float.hpp"
#include "arm/cmp.hpp"
#include "arm/div.hpp"
#include "arm/join.hpp"
#include "arm/load.hpp"
#include "arm/logical.hpp"
#include "arm/manip.hpp"
#include "arm/matrix.hpp"
#include "arm/minmax.hpp"
#include "arm/mul.hpp"
#include "arm/reciprocal.hpp"
#include "arm/rounding.hpp"
#include "arm/shift.hpp"
#include "arm/sub.hpp"
#include "arm/sqrt.hpp"
#endif

#endif // AMT_UI_ARCH_ARCH_HPP

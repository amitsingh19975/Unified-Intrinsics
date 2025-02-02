#ifndef AMT_UI_VEC_HEADERS_HPP
#define AMT_UI_VEC_HEADERS_HPP

#include "features.hpp"

#if UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
    #include <immintrin.h>
#elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE41
    #include <smmintrin.h>
#elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE1
    #include <xmmintrin.h>
#elif defined(UI_ARM_HAS_NEON)
    #include <arm_neon.h>
#elif defined(__wasm_simd128__)
    #include <wasm_simd128.h>
#endif

#endif // AMT_UI_VEC_HEADERS_HPP

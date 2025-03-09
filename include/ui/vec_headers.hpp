#ifndef AMT_UI_VEC_HEADERS_HPP
#define AMT_UI_VEC_HEADERS_HPP

#include "features.hpp"

#ifdef UI_EMPSCRIPTEN_WASM_SIMD
    #include <wasm_simd128.h>
#elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_AVX
    #include <immintrin.h>
#elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE41
    #include <smmintrin.h>
#elif UI_CPU_SSE_LEVEL >= UI_CPU_SSE_LEVEL_SSE1
    #include <xmmintrin.h>
#elif defined(UI_ARM_HAS_NEON)
    #include <arm_neon.h>
    #if defined __has_include
        #if __has_include(<arm_bf16.h>)
            #include <arm_bf16.h>
        #endif
    #endif
#endif

#endif // AMT_UI_VEC_HEADERS_HPP

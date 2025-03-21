#ifndef AMT_UI_FEATURES_HPP
#define AMT_UI_FEATURES_HPP

#include "base.hpp"

#if defined(__i386) || defined(_M_IX86) ||  defined(__x86_64__) || defined(_M_X64)
  #define UI_CPU_X86 1
#endif

#if defined(__powerpc__) || defined (__powerpc64__)
  #define UI_CPU_PPC 1
#endif

#define UI_CPU_SSE_LEVEL_SSE1     10
#define UI_CPU_SSE_LEVEL_SSE2     20
#define UI_CPU_SSE_LEVEL_SSE3     30
#define UI_CPU_SSE_LEVEL_SSSE3    31
#define UI_CPU_SSE_LEVEL_SSE41    41
#define UI_CPU_SSE_LEVEL_SSE42    42
#define UI_CPU_SSE_LEVEL_AVX      51
#define UI_CPU_SSE_LEVEL_AVX2     52
#define UI_CPU_SSE_LEVEL_SKX      60

#ifndef UI_CPU_SSE_LEVEL
    #if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512CD__) && \
        defined(__AVX512BW__) && defined(__AVX512VL__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SKX
    #elif defined(__AVX2__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_AVX2
    #elif defined(__AVX__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_AVX
    #elif defined(__SSE4_2__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SSE42
    #elif defined(__SSE4_1__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SSE41
    #elif defined(__SSSE3__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SSSE3
    #elif defined(__SSE3__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SSE3
    #elif defined(__SSE2__)
        #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SSE2
    #endif
#endif

#ifndef UI_CPU_SSE_LEVEL
    #if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512CD__) && \
        defined(__AVX512BW__) && defined(__AVX512VL__)
        #define UI_CPU_SSE_LEVEL        UI_CPU_SSE_LEVEL_SKX
    #elif defined(__AVX2__)
        #define UI_CPU_SSE_LEVEL        UI_CPU_SSE_LEVEL_AVX2
    #elif defined(__AVX__)
        #define UI_CPU_SSE_LEVEL        UI_CPU_SSE_LEVEL_AVX
    #elif defined(_M_X64) || defined(_M_AMD64)
        #define UI_CPU_SSE_LEVEL        UI_CPU_SSE_LEVEL_SSE2
    #elif defined(_M_IX86_FP)
        #if _M_IX86_FP >= 2
            #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SSE2
        #elif _M_IX86_FP == 1
            #define UI_CPU_SSE_LEVEL    UI_CPU_SSE_LEVEL_SSE1
        #endif
    #endif
#endif

#if defined(__arm__) && (!defined(__APPLE__) || !TARGET_IPHONE_SIMULATOR)
    #define UI_CPU_ARM32
#elif defined(__aarch64__)
    #define UI_CPU_ARM64
#endif

#if !defined(UI_ARM_HAS_NEON) && defined(__ARM_NEON)
    #define UI_ARM_HAS_NEON
#endif

#ifndef __is_identifier
  #define __is_identifier(x) 0
#endif

#define __has_keyword(__x) !(__is_identifier(__x))

#if defined __has_include
    #if __has_include(<stdfloat>)
        #include <stdfloat>
        #define UI_HAS_STD_FLOAT_HEADER
    #else
        #define __STDC_WANT_IEC_60559_TYPES_EXT__
        #include <float.h>
    #endif
#else
    #define __STDC_WANT_IEC_60559_TYPES_EXT__
    #include <float.h>
#endif

#ifndef UI_EMPSCRIPTEN
    #if defined(__STDCPP_FLOAT16_T__) || defined(FLT16_MIN) || __has_keyword(_Float16)
        #define UI_HAS_FLOAT_16
    #endif

    #if defined(__STDCPP_BFLOAT16_T__)
        #define UI_HAS_BFLOAT_16
    #endif
#endif

#ifndef UI_SUPPORT_FMA
    #if defined(__FMA__) || defined(__FMA4__)
        #define UI_SUPPORT_FMA
    #endif

    #if !defined(UI_SUPPORT_FMA) && defined(UI_COMPILER_MSVC) && defined(__AVX2__)
        #define UI_SUPPORT_FMA
    #endif
#endif

#ifdef __SIZEOF_INT128__
    #define UI_HAS_INT128
    namespace ui {
        using int128_t  = __int128_t;
        using uint128_t = __uint128_t;
    }
#endif

#ifndef UI_CACHE_LINE_SIZE
    #ifdef UI_ARM_HAS_NEON
        #define UI_CACHE_LINE_SIZE 128
    #elif defined(UI_COMPILER_MSVC)
      #if defined(_M_ARM) || defined(_M_ARM64)
        #define UI_CACHE_LINE_SIZE 64
      #elif defined(_M_X64) || defined(_M_IX86)
        #define UI_CACHE_LINE_SIZE 64
      #else
        #define UI_CACHE_LINE_SIZE 64  // default assumption
      #endif

    // GCC/Clang
    #elif defined(UI_COMPILER_GCC) || defined(UI_COMPILER_CLANG)
      #if defined(__aarch64__)
        #define UI_CACHE_LINE_SIZE 64
      #elif defined(__arm__)
        #define UI_CACHE_LINE_SIZE 32
      #elif defined(__x86_64__) || defined(__i386__)
        #define UI_CACHE_LINE_SIZE 64
      #else
        #define UI_CACHE_LINE_SIZE 64  // default fallback
      #endif

    #else
      #define UI_CACHE_LINE_SIZE 64  // default assumption
    #endif
#endif

#ifndef UI_USE_CSTDLIB
    #if !defined(UI_EMPSCRIPTEN)
        #define UI_USE_CSTDLIB
    #endif
#endif

#endif // AMT_UI_FEATURES_HPP

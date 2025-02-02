#ifndef AMT_UI_BASE_HPP
#define AMT_UI_BASE_HPP

#if !defined(UI_OS_ANDROID) && !defined(UI_OS_IOS) && !defined(UI_OS_WIN) && \
    !defined(UI_OS_UNIX) && !defined(UI_OS_MAC)

    #ifdef __APPLE__
        #include <TargetConditionals.h>
    #endif
    
    #if defined(_WIN32) || defined(__SYMBIAN32__)
        #define UI_OS_WIN
    #elif defined(ANDROID) || defined(__ANDROID__)
        #define UI_OS_ANDROID
    #elif defined(linux) || defined(__linux) || defined(__FreeBSD__) || \
          defined(__OpenBSD__) || defined(__sun) || defined(__NetBSD__) || \
          defined(__DragonFly__) || defined(__Fuchsia__) || \
          defined(__GLIBC__) || defined(__GNU__) || defined(__unix__)
        #define UI_OS_UNIX
    #elif TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        #define UI_OS_IOS
    #else
        #define UI_OS_MAC
    #endif
#endif

#if defined(__MSC_VER)
	#define UI_COMPILER_MSVC
#elif defined(__clang__)
	#define UI_COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
	#define UI_COMPILER_GCC
#endif

#ifdef UI_COMPILER_MSVC
    #define UI_ALWAYS_INLINE __forceinline inline
#else
    #define UI_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

#if defined(UI_COMPILER_CLANG)
    #define ASSUME(expr) __builtin_assume(expr)
#elif defined(UI_COMPILER_GCC)
    #define ASSUME(expr) if (expr) {} else { __builtin_unreachable(); }
#elif defined(UI_COMPILER_MSVC) || defined(__ICC)
    #define ASSUME(expr) __assume(expr)
#endif

#if !defined(UI_RESTRICT)
    #ifdef UI_COMPILER_MSVC
        #define UI_RESTRICT __restrict
    #else
        #define UI_RESTRICT __restrict__
    #endif
#endif

#endif // AMT_UI_BASE_HPP

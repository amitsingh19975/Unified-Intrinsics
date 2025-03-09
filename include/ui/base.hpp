#ifndef AMT_UI_BASE_HPP
#define AMT_UI_BASE_HPP

#ifndef UI_EMPSCRIPTEN
    #ifdef __EMSCRIPTEN__
        #define UI_EMPSCRIPTEN
    #endif
#endif

#ifndef UI_EMPSCRIPTEN_WASM_SIMD
    #if defined(UI_EMPSCRIPTEN) && defined(__wasm_simd128__)
        #define UI_EMPSCRIPTEN_WASM_SIMD
    #endif
#endif

#if !defined(UI_OS_ANDROID) && !defined(UI_OS_IOS) && !defined(UI_OS_WIN) && \
    !defined(UI_OS_UNIX) && !defined(UI_OS_MAC) && !defined(UI_EMPSCRIPTEN)

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


namespace ui {

    namespace op {
        template <unsigned char Op>
        struct OpTag {
            static constexpr auto id = Op;
        };
        using add_t                 = OpTag<0>; // +
        using sub_t                 = OpTag<1>; // -
        using mul_t                 = OpTag<2>; // *
        using div_t                 = OpTag<3>; // /
        using max_t                 = OpTag<4>;
        using min_t                 = OpTag<5>;

        // maximum operation avoiding NaN
        // max(4.f, NaN) == 4.f
        using maxnm_t               = OpTag<6>;
        // minimum operation avoiding NaN
        // min(4.f, NaN) == 4.f
        using minnm_t               = OpTag<7>;

        using and_test_t            = OpTag< 8>; // (lhs & rhs) != 0
        using equal_t               = OpTag< 9>; // =
        using equal_zero_t          = OpTag<10>; // =0

        using greater_t             = OpTag<11>; // >
        using greater_zero_t        = OpTag<12>; // >0
        using greater_equal_t       = OpTag<13>; // >=
        using greater_equal_zero_t  = OpTag<14>; // >=0
        using less_t                = OpTag<15>; // <
        using less_zero_t           = OpTag<16>; // <0
        using less_equal_t          = OpTag<17>; // <=
        using less_equal_zero_t     = OpTag<18>; // <=0

        using abs_greater_t         = OpTag<19>;
        using abs_greater_equal_t   = OpTag<20>;
        using abs_less_t            = OpTag<21>;
        using abs_less_equal_t      = OpTag<22>;

        using pmax_t                 = OpTag<23>; // pairwise max
        using pmin_t                 = OpTag<24>; // pairwise min
        using pmaxnm_t               = OpTag<25>; // pairwise maxnm
        using pminnm_t               = OpTag<26>; // pairwise minnm

        using padd_t                 = OpTag<27>; // pairwise add
    }

} // namespace ui

#endif // AMT_UI_BASE_HPP

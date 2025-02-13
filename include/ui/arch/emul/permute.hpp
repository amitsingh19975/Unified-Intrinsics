#ifndef AMT_UI_ARCH_PERMUTE_HPP
#define AMT_UI_ARCH_PERMUTE_HPP

#include "cast.hpp"
#include "ui/maths.hpp"

namespace ui {

    namespace internal {
        #if defined(UI_COMPILER_CLANG) || defined(UI_COMPILER_GCC)
            #if defined(UI_COMPILER_CLANG)
                template <std::size_t N, typename T>
                using VecExt = T __attribute__((ext_vector_type(N)));
            #else
                namespace impl {
                    template <std::size_t N, typename T>
                    struct VecExtHelper {
                        typedef T __attribute__((vector_size(N * sizeof(T)))) type;
                    };
                }

                template <std::size_t N, typename T>
                using VecExt = typename impl::VecExtHelper<N, T>::type;
            #endif
            template <std::size_t N, typename T>
            static inline constexpr auto to_vext(Vec<N, T> const& v) noexcept -> VecExt<N, T> {
                return std::bit_cast<VecExt<N, T>>(v);
            }
            #define UI_VECTOR_EXTENSION
        #endif
    } // namespace internal

    template <std::size_t... Is, std::size_t N, typename T>
        requires (maths::is_power_of_2(sizeof...(Is)))
    UI_ALWAYS_INLINE static constexpr auto shuffle(
        Vec<N, T> const& x
    ) noexcept -> Vec<sizeof...(Is), T> {
        using namespace internal;
        static constexpr auto R = sizeof...(Is);
        #if defined(UI_COMPILER_CLANG) || defined(UI_COMPILER_GCC)
            return std::bit_cast<Vec<R, T>>(__builtin_shufflevector(to_vext(x), to_vext(x), Is...));
        #else
            return Vec<R, T>::load(x[Is]...);
        #endif
    }
} // ui

#endif // AMT_UI_ARCH_PERMUTE_HPP 

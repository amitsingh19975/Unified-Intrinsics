#ifndef AMT_UI_ARCH_X86_BASIC_HPP
#define AMT_UI_ARCH_X86_BASIC_HPP

namespace ui {
    namespace constants {
        alignas(16) static constexpr int8_t mask8_16_even_odd[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7,  9, 11, 13, 15 };
        alignas(16) static constexpr int8_t mask8_32_even_odd[16] = { 0, 1, 4, 5, 8,  9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 };
    }
}

#endif // AMT_UI_ARCH_X86_BASIC_HPP

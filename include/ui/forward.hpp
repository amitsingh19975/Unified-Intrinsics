#ifndef AMT_UI_FORWARD_HPP
#define AMT_UI_FORWARD_HPP

#include <cstddef>
#include <cstdint>

namespace ui {

    template<std::size_t N, typename T>
    struct alignas(N * sizeof(T)) Vec;
    
    struct alignas(sizeof(std::uint16_t)) float16;
    
    struct alignas(sizeof(std::uint16_t)) bfloat16;
} // namespace ui

#endif // AMT_UI_FORWARD_HPP

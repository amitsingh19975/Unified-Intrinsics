#ifndef AMT_UI_MATHS_HPP
#define AMT_UI_MATHS_HPP

#include <type_traits>

namespace ui::maths {

    template <typename T>
        requires std::is_arithmetic_v<T>
    static inline constexpr auto is_power_of_2(T num) noexcept -> bool {
        if (num == 0) return true;
        return (num & (num - 1)) == 0;
    }

} // namespace ui::maths

#endif // AMT_UI_MATHS_HPP

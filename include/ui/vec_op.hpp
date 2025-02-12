#ifndef AMT_UI_VEC_OP_HPP
#define AMT_UI_VEC_OP_HPP

#include "base_vec.hpp"
#include "features.hpp"
#include <type_traits>
#include <utility>
#include "float.hpp"
#include "arch/arch.hpp"
#include "ui/arch/arm/shift.hpp"
#include "ui/base.hpp"

namespace ui {
    template <std::size_t N, typename T>
        requires (std::is_arithmetic_v<T>)
    static inline constexpr auto load(T val) noexcept -> Vec<N, T> {
        #if defined(UI_ARM_HAS_NEON)
            return arm::neon::load<N>(val);
        #else
            #error "not implemented"
        #endif
    }

    template <std::size_t N, typename T>
    inline constexpr auto Vec<N, T>::load(T val) noexcept -> Vec<N, T> {
        return ui::load<N, T>(val);
    }

    template <std::size_t N, unsigned Lane, std::size_t M, typename T>
    inline constexpr auto load(Vec<M, T> const& v) noexcept -> Vec<N, T> {
        #if defined(UI_ARM_HAS_NEON)
            return arm::neon::load<N, Lane>(v);
        #else
            #error "not implemented"
        #endif
    }

    template <std::size_t N, typename T>
    template <unsigned Lane, std::size_t M>
    inline constexpr auto Vec<N, T>::load(Vec<M, T> const& v) noexcept -> Vec<N, T> {
        return ui::load<N, Lane>(v);
    }
} // namespace ui

// MARK: not
template <std::size_t N, std::integral T>
    requires (!std::is_signed_v<T>)
UI_ALWAYS_INLINE constexpr auto operator!(ui::Vec<N, T> const& op) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_not(op);
}

template <std::size_t N, std::integral T>
    requires (!std::is_signed_v<T>)
UI_ALWAYS_INLINE constexpr auto operator~(ui::Vec<N, T> const& op) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_not(op);
}
// !MARK

// MARK: ==
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator==(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::equal_t{});
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator==(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::mask_t<N, T> {
    return lhs == ui::Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator==(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    return rhs == lhs;
}
// !MARK

// MARK: !=
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator!=(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return !cmp(lhs, rhs, op::equal_t{});
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator!=(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::mask_t<N, T> {
    return lhs != ui::Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator!=(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    return !(lhs == rhs);
}
// !MARK

// MARK: <=
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<=(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::less_equal_t{});
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<=(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs <= Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<=(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) <= rhs;
}
// !MARK

// MARK: <
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::less_t{});
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs < Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator<(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) < rhs;
}
// !MARK

// MARK: >=
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>=(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::greater_equal_t{});
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>=(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs >= Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>=(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) >= rhs;
}
// !MARK

// MARK: >
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return cmp(lhs, rhs, op::greater_t{});
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs > Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator>(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) > rhs;
}
// !MARK

// MARK: - (negate)
template <std::size_t N, typename T>
    requires (std::floating_point<T> || std::is_signed_v<T>)
UI_ALWAYS_INLINE constexpr auto operator-(ui::Vec<N, T> const& op) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return negate(op);
}
// !MARK

// MARK: ^
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator^(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_xor(lhs, rhs);
}

template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator^(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return lhs ^ Vec<N, T>::load(rhs);
}

template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator^(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::mask_t<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) ^ rhs;
}
// !MARK

// MARK: +
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator+(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return add(lhs, rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator+(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs + Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator+(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) + rhs;
}
// !MARK

// MARK: -
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator-(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return sub(lhs, rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator-(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs - Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator-(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) - rhs;
}
// !MARK

// MARK: *
template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator*(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return mul(lhs, rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator*(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs * Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator*(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) * rhs;
}
// !MARK


// MARK: /
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator/(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return div(lhs, rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator/(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs / Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator/(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) / rhs;
}
// !MARK

// MARK: &
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator&(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_and(lhs, rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator&(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs & Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator&(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) & rhs;
}
// !MARK

// MARK: |
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator|(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return bitwise_or(lhs, rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator|(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs | Vec<N, T>::load(rhs);
}

template <std::size_t N, typename T>
UI_ALWAYS_INLINE constexpr auto operator|(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) | rhs;
}
// !MARK

// MARK: <<
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator<<(ui::Vec<N, T> const& lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return shift_left(lhs, rhs);
}

template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator<<(ui::Vec<N, T> const& lhs, T const rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return lhs << Vec<N, T>::load(rhs);
}

template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator<<(T const lhs, ui::Vec<N, T> const& rhs) noexcept -> ui::Vec<N, T> {
    using namespace ui;
    return Vec<N, T>::load(lhs) << rhs;
}
// !MARK

// MARK: >>
namespace ui::internal {
    template <unsigned I, std::size_t N, typename T>
    UI_ALWAYS_INLINE constexpr auto shift_right_helper(Vec<N, T> const& lhs, unsigned const rhs) noexcept -> Vec<N, T> {
        if constexpr (I < sizeof(T) * 8) {
            switch (rhs) {
                case I + 0: {
                    if constexpr (I != 0) {
                        return ::ui::shift_right<I + 0>(lhs);
                    } else return lhs;
                }
                case I + 1: {
                    if constexpr (I + 1 < sizeof(T) * 8) {
                        return ::ui::shift_right<I + 1>(lhs);
                    }
                    break;
                }
                case I + 2: {
                    if constexpr (I + 2 < sizeof(T) * 8) {
                        return ::ui::shift_right<I + 2>(lhs);
                    }
                    break;
                }
                case I + 3: {
                    if constexpr (I + 3 < sizeof(T) * 8) {
                        return ::ui::shift_right<I + 3>(lhs);
                    }
                    break;
                }
                case I + 4: {
                    if constexpr (I + 4 < sizeof(T) * 8) {
                        return ::ui::shift_right<I + 4>(lhs);
                    }
                    break;
                }
                case I + 5: {
                    if constexpr (I + 5 < sizeof(T) * 8) {
                        return ::ui::shift_right<I + 5>(lhs);
                    }
                    break;
                }
                case I + 6: {
                    if constexpr (I + 6 < sizeof(T) * 8) {
                        return ::ui::shift_right<I + 6>(lhs);
                    }
                    break;
                }
                case I + 7: {
                    if constexpr (I + 7 < sizeof(T) * 8) {
                        return ::ui::shift_right<I + 7>(lhs);
                    }
                    break;
                }
                default: return shift_right_helper<I + 8>(lhs, rhs);
            }
        }
        return {};
    }
} // namespace ui:internal
template <std::size_t N, std::integral T>
UI_ALWAYS_INLINE constexpr auto operator>>(ui::Vec<N, T> const& lhs, unsigned const rhs) noexcept -> ui::Vec<N, T> {
    return ui::internal::shift_right_helper<0>(lhs, rhs);
}
// !MARK
#endif // AMT_UI_VEC_OP_HPP

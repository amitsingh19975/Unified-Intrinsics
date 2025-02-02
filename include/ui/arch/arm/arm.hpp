#ifndef AMT_UI_ARCH_ARM_ARM_HPP
#define AMT_UI_ARCH_ARM_ARM_HPP

#include "cast.hpp"
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace ui::arm {

    template <std::size_t N, typename T>
    UI_ALWAYS_INLINE auto load(T val) noexcept -> VecReg<N, T> {
		using ret_t = VecReg<N, T>;
		
		if constexpr (N == 1) {
			return { .val = val };
		} else if constexpr (sizeof(T) == 1) {
			if constexpr (N == 8)
				return std::bit_cast<ret_t>(vdup_n_u8(std::bit_cast<std::uint8_t>(val)));
			else if constexpr (N == 16)
				return std::bit_cast<ret_t>(vdupq_n_u8(std::bit_cast<std::uint8_t>(val)));
		} else if constexpr (sizeof(T) == 2) {
			if constexpr (N == 4)
				return std::bit_cast<ret_t>(vdup_n_u16(std::bit_cast<std::uint16_t>(val)));
			else if constexpr (N == 8)
				return std::bit_cast<ret_t>(vdupq_n_u16(std::bit_cast<std::uint16_t>(val)));
		} else if constexpr (sizeof(T) == 4) {
			if constexpr (N == 2)
				return std::bit_cast<ret_t>(vdup_n_u32(std::bit_cast<std::uint32_t>(val)));
			else if constexpr (N == 4)
				return std::bit_cast<ret_t>(vdupq_n_u32(std::bit_cast<std::uint32_t>(val)));
		} else if constexpr (sizeof(T) == 8) {
			if constexpr (N == 1)
				return std::bit_cast<ret_t>(vdup_n_u64(std::bit_cast<std::uint64_t>(val)));
			else if constexpr (N == 2)
				return std::bit_cast<ret_t>(vdupq_n_u64(std::bit_cast<std::uint64_t>(val)));
		}

		if constexpr (N > 1) {
			return join(load<N/2>(val), load<N/2>(val));
		}
    }

} // namespace ui::arm;

#include "add.hpp"
#include "sub.hpp"
#include "abs.hpp"
#include "minmax.hpp"
#include "mul.hpp"
#include "div.hpp"
#include "rounding.hpp"
#include "reciprocal.hpp"

#endif // AMT_UI_ARCH_ARM_ARM_HPP

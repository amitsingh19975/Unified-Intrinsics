#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <print>
#include "ui.hpp"
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace ui;

template <typename U>
struct ImplicitCast {
	U data;

	template <typename T>
	constexpr operator T() const noexcept {
		return static_cast<T>(data);
	}
};

template <typename U>
ImplicitCast(U) -> ImplicitCast<U>;

template <typename T>
struct is_implicit_cast_helper: std::false_type{};

template <typename T>
struct is_implicit_cast_helper<ImplicitCast<T>>: std::true_type{};

template <typename T>
concept is_implicit_cast = is_implicit_cast_helper<T>::value;

template <typename T>
constexpr auto operator==(T lhs, is_implicit_cast auto rhs) noexcept {
	return lhs == static_cast<T>(rhs.data);
}

template <typename T>
constexpr auto operator!=(T lhs, is_implicit_cast auto rhs) noexcept {
	return !(lhs == rhs);
}

template <typename T>
constexpr auto operator==(is_implicit_cast auto lhs, T rhs) noexcept {
	return static_cast<T>(lhs.data) == rhs;
}

template <typename T>
constexpr auto operator!=(is_implicit_cast auto lhs, T rhs) noexcept {
	return !(lhs == rhs);
}

template <std::floating_point T = float>
static constexpr T eps = T(0.1);

TEST_CASE( VEC_ARCH_NAME " Casting From 8bit integer", "[from_8bit_int]" ) {
	GIVEN("An unsigned 8bit integer") {
		auto v = Vec<16, std::uint8_t>::load(
			0, UINT8_MAX, 3, 4, 5, 6, 7, 8,
			9, 10, 11, 12, 13, 14, 15, 16
		);

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == 0);
			REQUIRE(v[1]  == UINT8_MAX);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
			REQUIRE(v[4]  == 5);
			REQUIRE(v[5]  == 6);
			REQUIRE(v[6]  == 7);
			REQUIRE(v[7]  == 8);
			REQUIRE(v[8]  == 9);
			REQUIRE(v[9]  == 10);
			REQUIRE(v[10] == 11);
			REQUIRE(v[11] == 12);
			REQUIRE(v[12] == 13);
			REQUIRE(v[13] == 14);
			REQUIRE(v[14] == 15);
			REQUIRE(v[15] == 16);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			auto m = static_cast<int8_t>(UINT8_MAX);
			REQUIRE(res[1]  == m);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		/*WHEN("Casting to float16 integer") {*/
		/*	auto res = cast<float16>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, float16>);*/
		/**/
		/*	REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0	     , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(UINT8_MAX, eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 8]), Catch::Matchers::WithinRel(9        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 9]), Catch::Matchers::WithinRel(10       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[10]), Catch::Matchers::WithinRel(11       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[11]), Catch::Matchers::WithinRel(12       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[12]), Catch::Matchers::WithinRel(13       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[13]), Catch::Matchers::WithinRel(14       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[14]), Catch::Matchers::WithinRel(15       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[15]), Catch::Matchers::WithinRel(16       , eps<float>));*/
		/*}*/
		/**/
		/*WHEN("Casting to bfloat16 integer") {*/
		/*	auto res = cast<bfloat16>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
		/**/
		/*	REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0	     , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(UINT8_MAX, eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 8]), Catch::Matchers::WithinRel(9        , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 9]), Catch::Matchers::WithinRel(10       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[10]), Catch::Matchers::WithinRel(11       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[11]), Catch::Matchers::WithinRel(12       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[12]), Catch::Matchers::WithinRel(13       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[13]), Catch::Matchers::WithinRel(14       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[14]), Catch::Matchers::WithinRel(15       , eps<float>));*/
		/*	REQUIRE_THAT(float(res[15]), Catch::Matchers::WithinRel(16       , eps<float>));*/
		/*}*/
		/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0	      , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(UINT8_MAX, eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3        , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4        , eps<float>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5        , eps<float>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6        , eps<float>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7        , eps<float>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8        , eps<float>));
			REQUIRE_THAT(res[ 8], Catch::Matchers::WithinRel(9        , eps<float>));
			REQUIRE_THAT(res[ 9], Catch::Matchers::WithinRel(10       , eps<float>));
			REQUIRE_THAT(res[10], Catch::Matchers::WithinRel(11       , eps<float>));
			REQUIRE_THAT(res[11], Catch::Matchers::WithinRel(12       , eps<float>));
			REQUIRE_THAT(res[12], Catch::Matchers::WithinRel(13       , eps<float>));
			REQUIRE_THAT(res[13], Catch::Matchers::WithinRel(14       , eps<float>));
			REQUIRE_THAT(res[14], Catch::Matchers::WithinRel(15       , eps<float>));
			REQUIRE_THAT(res[15], Catch::Matchers::WithinRel(16       , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0	      , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(UINT8_MAX, eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3        , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4        , eps<double>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5        , eps<double>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6        , eps<double>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7        , eps<double>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8        , eps<double>));
			REQUIRE_THAT(res[ 8], Catch::Matchers::WithinRel(9        , eps<double>));
			REQUIRE_THAT(res[ 9], Catch::Matchers::WithinRel(10       , eps<double>));
			REQUIRE_THAT(res[10], Catch::Matchers::WithinRel(11       , eps<double>));
			REQUIRE_THAT(res[11], Catch::Matchers::WithinRel(12       , eps<double>));
			REQUIRE_THAT(res[12], Catch::Matchers::WithinRel(13       , eps<double>));
			REQUIRE_THAT(res[13], Catch::Matchers::WithinRel(14       , eps<double>));
			REQUIRE_THAT(res[14], Catch::Matchers::WithinRel(15       , eps<double>));
			REQUIRE_THAT(res[15], Catch::Matchers::WithinRel(16       , eps<double>));
		}


		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = sat_cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

	}

	GIVEN("A signed 8bit integer") {
		auto v = Vec<16, std::int8_t>::load(
			INT8_MIN, INT8_MAX, 3, 4, 5, 6, 7, 8,
			9, 10, 11, 12, 13, 14, 15, 16
		);

		auto min = ImplicitCast(INT8_MIN);
		auto max = ImplicitCast(INT8_MAX);

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == min);
			REQUIRE(v[1]  == max);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
			REQUIRE(v[4]  == 5);
			REQUIRE(v[5]  == 6);
			REQUIRE(v[6]  == 7);
			REQUIRE(v[7]  == 8);
			REQUIRE(v[8]  == 9);
			REQUIRE(v[9]  == 10);
			REQUIRE(v[10] == 11);
			REQUIRE(v[11] == 12);
			REQUIRE(v[12] == 13);
			REQUIRE(v[13] == 14);
			REQUIRE(v[14] == 15);
			REQUIRE(v[15] == 16);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}


		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

	/*	WHEN("Casting to float16 integer") {*/
	/*		auto res = cast<float16>(v);*/
	/*		static_assert(std::same_as<decltype(res)::element_t, float16>);*/
	/**/
	/*		REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(INT8_MIN , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INT8_MAX , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 8]), Catch::Matchers::WithinRel(9        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 9]), Catch::Matchers::WithinRel(10       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[10]), Catch::Matchers::WithinRel(11       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[11]), Catch::Matchers::WithinRel(12       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[12]), Catch::Matchers::WithinRel(13       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[13]), Catch::Matchers::WithinRel(14       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[14]), Catch::Matchers::WithinRel(15       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[15]), Catch::Matchers::WithinRel(16       , eps<float>));*/
	/*	}*/
	/**/
	/*	WHEN("Casting to bfloat16 integer") {*/
	/*		auto res = cast<bfloat16>(v);*/
	/*		static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
	/**/
	/*		REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(INT8_MIN , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INT8_MAX , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 8]), Catch::Matchers::WithinRel(9        , eps<float>));*/
	/*		REQUIRE_THAT(float(res[ 9]), Catch::Matchers::WithinRel(10       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[10]), Catch::Matchers::WithinRel(11       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[11]), Catch::Matchers::WithinRel(12       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[12]), Catch::Matchers::WithinRel(13       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[13]), Catch::Matchers::WithinRel(14       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[14]), Catch::Matchers::WithinRel(15       , eps<float>));*/
	/*		REQUIRE_THAT(float(res[15]), Catch::Matchers::WithinRel(16       , eps<float>));*/
	/*	}*/
	/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT8_MIN , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT8_MAX , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3        , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4        , eps<float>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5        , eps<float>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6        , eps<float>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7        , eps<float>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8        , eps<float>));
			REQUIRE_THAT(res[ 8], Catch::Matchers::WithinRel(9        , eps<float>));
			REQUIRE_THAT(res[ 9], Catch::Matchers::WithinRel(10       , eps<float>));
			REQUIRE_THAT(res[10], Catch::Matchers::WithinRel(11       , eps<float>));
			REQUIRE_THAT(res[11], Catch::Matchers::WithinRel(12       , eps<float>));
			REQUIRE_THAT(res[12], Catch::Matchers::WithinRel(13       , eps<float>));
			REQUIRE_THAT(res[13], Catch::Matchers::WithinRel(14       , eps<float>));
			REQUIRE_THAT(res[14], Catch::Matchers::WithinRel(15       , eps<float>));
			REQUIRE_THAT(res[15], Catch::Matchers::WithinRel(16       , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT8_MIN , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT8_MAX , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3        , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4        , eps<double>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5        , eps<double>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6        , eps<double>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7        , eps<double>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8        , eps<double>));
			REQUIRE_THAT(res[ 8], Catch::Matchers::WithinRel(9        , eps<double>));
			REQUIRE_THAT(res[ 9], Catch::Matchers::WithinRel(10       , eps<double>));
			REQUIRE_THAT(res[10], Catch::Matchers::WithinRel(11       , eps<double>));
			REQUIRE_THAT(res[11], Catch::Matchers::WithinRel(12       , eps<double>));
			REQUIRE_THAT(res[12], Catch::Matchers::WithinRel(13       , eps<double>));
			REQUIRE_THAT(res[13], Catch::Matchers::WithinRel(14       , eps<double>));
			REQUIRE_THAT(res[14], Catch::Matchers::WithinRel(15       , eps<double>));
			REQUIRE_THAT(res[15], Catch::Matchers::WithinRel(16       , eps<double>));
		}

		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = sat_cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
			REQUIRE(res[8]  == 9);
			REQUIRE(res[9]  == 10);
			REQUIRE(res[10] == 11);
			REQUIRE(res[11] == 12);
			REQUIRE(res[12] == 13);
			REQUIRE(res[13] == 14);
			REQUIRE(res[14] == 15);
			REQUIRE(res[15] == 16);
		}

	}
}

TEST_CASE( "Casting From 16bit integer", "[from_16bit_int]" ) {
	GIVEN("An unsigned 16bit integer") {
		auto v = Vec<8, std::uint16_t>::load(
			0, UINT16_MAX, 3, 4, 5, 6, 7, 8
		);

		auto max = ImplicitCast{ UINT16_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == 0);
			REQUIRE(v[1]  == max);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
			REQUIRE(v[4]  == 5);
			REQUIRE(v[5]  == 6);
			REQUIRE(v[6]  == 7);
			REQUIRE(v[7]  == 8);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8   , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(max , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8   , eps<float>));*/
/*		}*/
/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0   , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(max , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3   , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4   , eps<float>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5   , eps<float>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6   , eps<float>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7   , eps<float>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8   , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0   , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(max , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3   , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4   , eps<double>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5   , eps<double>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6   , eps<double>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7   , eps<double>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8   , eps<double>));
		}


		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}
	}

	GIVEN("A signed 16bit integer") {
		auto v = Vec<8, std::int16_t>::load(
			INT16_MIN, INT16_MAX, 3, 4, 5, 6, 7, 8
		);

		auto min = ImplicitCast{ INT16_MIN };
		auto max = ImplicitCast{ INT16_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == min);
			REQUIRE(v[1]  == max);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
			REQUIRE(v[4]  == 5);
			REQUIRE(v[5]  == 6);
			REQUIRE(v[6]  == 7);
			REQUIRE(v[7]  == 8);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(INT16_MIN , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INT16_MAX , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8         , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(INT16_MIN , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INT16_MAX , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 4]), Catch::Matchers::WithinRel(5         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 5]), Catch::Matchers::WithinRel(6         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 6]), Catch::Matchers::WithinRel(7         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 7]), Catch::Matchers::WithinRel(8         , eps<float>));*/
/*		}*/
/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT16_MIN , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT16_MAX , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3         , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4         , eps<float>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5         , eps<float>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6         , eps<float>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7         , eps<float>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8         , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT16_MIN , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT16_MAX , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3         , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4         , eps<double>));
			REQUIRE_THAT(res[ 4], Catch::Matchers::WithinRel(5         , eps<double>));
			REQUIRE_THAT(res[ 5], Catch::Matchers::WithinRel(6         , eps<double>));
			REQUIRE_THAT(res[ 6], Catch::Matchers::WithinRel(7         , eps<double>));
			REQUIRE_THAT(res[ 7], Catch::Matchers::WithinRel(8         , eps<double>));
		}


		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == INT8_MIN);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == INT16_MIN);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
			REQUIRE(res[4]  == 5);
			REQUIRE(res[5]  == 6);
			REQUIRE(res[6]  == 7);
			REQUIRE(res[7]  == 8);
		}
	}
}

TEST_CASE( "Casting From 32bit integer", "[from_32bit_int]" ) {
	GIVEN("An unsigned 32bit integer") {
		auto v = Vec<4, std::uint32_t>::load(
			0, UINT32_MAX, 3, 4
		);

		auto max = ImplicitCast{ UINT32_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == 0);
			REQUIRE(v[1]  == max);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4   , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(max , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4   , eps<float>));*/
/*		}*/
/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0   , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(max , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3   , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4   , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0   , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(max , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3   , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4   , eps<double>));
		}


		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}
	}

	GIVEN("A signed 32bit integer") {
		auto v = Vec<4, std::int32_t>::load(
			INT32_MIN, INT32_MAX, 3, 4
		);

		auto min = ImplicitCast{ INT32_MIN };
		auto max = ImplicitCast{ INT32_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == min);
			REQUIRE(v[1]  == max);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INFINITY  , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4         , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(INT32_MIN , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INT32_MAX , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4         , eps<float>));*/
/*		}*/
/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT32_MIN , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT32_MAX , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3         , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4         , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT32_MIN , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT32_MAX , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3         , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4         , eps<double>));
		}

		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == INT8_MIN);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == INT16_MIN);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}
	}
}

TEST_CASE( "Casting From 64bit integer", "[from_64bit_int]" ) {
	GIVEN("An unsigned 64bit integer") {
		auto v = Vec<4, std::uint64_t>::load(
			0, UINT64_MAX, 3, 4
		);

		auto max = ImplicitCast{ UINT64_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == 0);
			REQUIRE(v[1]  == max);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0	     , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3   	 , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4        , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(0   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(max , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3   , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4   , eps<float>));*/
/*		}*/
/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0   , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(max , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3   , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4   , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(0   , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(max , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3   , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4   , eps<double>));
		}

		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}
	}

	GIVEN("A signed 64bit integer") {
		auto v = Vec<4, std::int64_t>::load(
			INT64_MIN, INT64_MAX, 3, 4
		);

		auto min = ImplicitCast{ INT64_MIN };
		auto max = ImplicitCast{ INT64_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE(v[0]  == min);
			REQUIRE(v[1]  == max);
			REQUIRE(v[2]  == 3);
			REQUIRE(v[3]  == 4);
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == min);
			REQUIRE(res[1]  == max);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4         , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(INT64_MIN , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel(INT64_MAX , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3         , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4         , eps<float>));*/
/*		}*/
/**/
		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT64_MIN , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT64_MAX , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3         , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4         , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(INT64_MIN , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(INT64_MAX , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3         , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4         , eps<double>));
		}

		WHEN("Saturating casting to signed 8bit integer") {
			auto res = sat_cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == INT8_MIN);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 16bit integer") {
			auto res = sat_cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == INT16_MIN);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 16bit integer") {
			auto res = sat_cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 32bit integer") {
			auto res = sat_cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == INT32_MIN);
			REQUIRE(res[1]  == INT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 32bit integer") {
			auto res = sat_cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to signed 64bit integer") {
			auto res = sat_cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == INT64_MIN);
			REQUIRE(res[1]  == INT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Saturating casting to unsigned 64bit integer") {
			auto res = sat_cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == INT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}
	}
}

/*TEST_CASE( "Casting From 16bit float", "[from_16bit_float]" ) {*/
/*	GIVEN("A 16bit float") {*/
/*		auto v = Vec<4, float16>::load(*/
/*			-INFINITY, INFINITY, 3.12, 4.12*/
/*		);*/
/**/
/*		auto min = ImplicitCast{ INT64_MIN };*/
/*		auto max = ImplicitCast{ INT64_MAX };*/
/**/
/*		THEN("The vector elements are the same as provided") {*/
/*			REQUIRE_THAT(float(v[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(v[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(v[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(float(v[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to signed 8bit integer") {*/
/*			auto res = cast<int8_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int8_t>);*/
/*			REQUIRE(res[0]  == INT8_MIN);*/
/*			REQUIRE(res[1]  == INT8_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 8bit integer") {*/
/*			auto res = cast<uint8_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT8_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to signed 16bit integer") {*/
/*			auto res = cast<int16_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int16_t>);*/
/*			REQUIRE(res[0]  == INT16_MIN);*/
/*			REQUIRE(res[1]  == INT16_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 16bit integer") {*/
/*			auto res = cast<uint16_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT16_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to signed 32bit integer") {*/
/*			auto res = cast<int32_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int32_t>);*/
/*			REQUIRE(res[0]  == INT32_MIN);*/
/*			REQUIRE(res[1]  == INT32_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 32bit integer") {*/
/*			auto res = cast<uint32_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT32_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to signed 64bit integer") {*/
/*			auto res = cast<int64_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int64_t>);*/
/*			REQUIRE(res[0]  == INT64_MIN);*/
/*			REQUIRE(res[1]  == INT64_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 64bit integer") {*/
/*			auto res = cast<uint64_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT64_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to float32 integer") {*/
/*			auto res = cast<float>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float>);*/
/**/
/*			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to float64 integer") {*/
/*			auto res = cast<double>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, double>);*/
/**/
/*			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(double(-INFINITY) , eps<double>));*/
/*			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(double( INFINITY) , eps<double>));*/
/*			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12      , eps<double>));*/
/*			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12      , eps<double>));*/
/*		}*/
/*	}*/
/*}*/
/**/
/*TEST_CASE( "Casting From 16bit brain-float", "[from_16bit_bfloat]" ) {*/
/*	GIVEN("A 16bit bfloat") {*/
/*		auto v = Vec<4, bfloat16>::load(*/
/*			-INFINITY, INFINITY, 3.12, 4.12*/
/*		);*/
/**/
/*		auto min = ImplicitCast{ INT64_MIN };*/
/*		auto max = ImplicitCast{ INT64_MAX };*/
/**/
/*		THEN("The vector elements are the same as provided") {*/
/*			REQUIRE_THAT(float(v[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(v[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(v[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(float(v[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to signed 8bit integer") {*/
/*			auto res = cast<int8_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int8_t>);*/
/*			REQUIRE(res[0]  == INT8_MIN);*/
/*			REQUIRE(res[1]  == INT8_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 8bit integer") {*/
/*			auto res = cast<uint8_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT8_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to signed 16bit integer") {*/
/*			auto res = cast<int16_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int16_t>);*/
/*			REQUIRE(res[0]  == INT16_MIN);*/
/*			REQUIRE(res[1]  == INT16_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 16bit integer") {*/
/*			auto res = cast<uint16_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT16_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to signed 32bit integer") {*/
/*			auto res = cast<int32_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int32_t>);*/
/*			REQUIRE(res[0]  == INT32_MIN);*/
/*			REQUIRE(res[1]  == INT32_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 32bit integer") {*/
/*			auto res = cast<uint32_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT32_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to signed 64bit integer") {*/
/*			auto res = cast<int64_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, int64_t>);*/
/*			REQUIRE(res[0]  == INT64_MIN);*/
/*			REQUIRE(res[1]  == INT64_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to unsigned 64bit integer") {*/
/*			auto res = cast<uint64_t>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);*/
/*			REQUIRE(res[0]  == 0);*/
/*			REQUIRE(res[1]  == UINT64_MAX);*/
/*			REQUIRE(res[2]  == 3);*/
/*			REQUIRE(res[3]  == 4);*/
/*		}*/
/**/
/*		WHEN("Casting to float16 integer") {*/
/*			auto res = cast<float16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to bfloat16 integer") {*/
/*			auto res = cast<bfloat16>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
/**/
/*			REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to float32 integer") {*/
/*			auto res = cast<float>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, float>);*/
/**/
/*			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
/*			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
/*			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
/*			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
/*		}*/
/**/
/*		WHEN("Casting to float64 integer") {*/
/*			auto res = cast<double>(v);*/
/*			static_assert(std::same_as<decltype(res)::element_t, double>);*/
/**/
/*			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(double(-INFINITY) , eps<double>));*/
/*			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(double( INFINITY) , eps<double>));*/
/*			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12      , eps<double>));*/
/*			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12      , eps<double>));*/
/*		}*/
/*	}*/
/*}*/
/**/
TEST_CASE( VEC_ARCH_NAME" Casting From 32bit float", "[from_32bit_float]" ) {
	GIVEN("A 32bit float") {
		auto v = Vec<4, float>::load(
			-INFINITY, INFINITY, 3.12, 4.12
		);

		auto min = ImplicitCast{ INT64_MIN };
		auto max = ImplicitCast{ INT64_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE_THAT(v[ 0], Catch::Matchers::WithinRel(-INFINITY , eps<float>));
			REQUIRE_THAT(v[ 1], Catch::Matchers::WithinRel( INFINITY , eps<float>));
			REQUIRE_THAT(v[ 2], Catch::Matchers::WithinRel(3.12f     , eps<float>));
			REQUIRE_THAT(v[ 3], Catch::Matchers::WithinRel(4.12f     , eps<float>));
		}

		WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == INT8_MIN);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == INT16_MIN);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == INT32_MIN);
			REQUIRE(res[1]  == INT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == INT64_MIN);
			REQUIRE(res[1]  == INT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		/*WHEN("Casting to float16 integer") {*/
		/*	auto res = cast<float16>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, float16>);*/
		/**/
		/*	REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
		/*}*/
		/**/
		/*WHEN("Casting to bfloat16 integer") {*/
		/*	auto res = cast<bfloat16>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
		/**/
		/*	REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
		/*}*/

		WHEN("Casting to float32 integer") {
			auto res = cast<float>(v);
			static_assert(std::same_as<decltype(res)::element_t, float>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(-INFINITY , eps<float>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel( INFINITY , eps<float>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12f     , eps<float>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12f     , eps<float>));
		}

		WHEN("Casting to float64 integer") {
			auto res = cast<double>(v);
			static_assert(std::same_as<decltype(res)::element_t, double>);

			REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(double(-INFINITY) , eps<double>));
			REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(double( INFINITY) , eps<double>));
			REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12      , eps<double>));
			REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12      , eps<double>));
		}
	}
}

TEST_CASE( "Casting From 64bit float", "[from_64bit_float]" ) {
	GIVEN("A 32bit float") {
		auto v = Vec<4, double>::load(
			-INFINITY, INFINITY, 3.12, 4.12
		);

		auto min = ImplicitCast{ INT64_MIN };
		auto max = ImplicitCast{ INT64_MAX };

		THEN("The vector elements are the same as provided") {
			REQUIRE_THAT(v[ 0], Catch::Matchers::WithinRel(double(-INFINITY) , eps<double>));
			REQUIRE_THAT(v[ 1], Catch::Matchers::WithinRel(double( INFINITY) , eps<double>));
			REQUIRE_THAT(v[ 2], Catch::Matchers::WithinRel(3.12      , eps<double>));
			REQUIRE_THAT(v[ 3], Catch::Matchers::WithinRel(4.12      , eps<double>));
		}

			WHEN("Casting to signed 8bit integer") {
			auto res = cast<int8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int8_t>);
			REQUIRE(res[0]  == INT8_MIN);
			REQUIRE(res[1]  == INT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 8bit integer") {
			auto res = cast<uint8_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint8_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT8_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 16bit integer") {
			auto res = cast<int16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int16_t>);
			REQUIRE(res[0]  == INT16_MIN);
			REQUIRE(res[1]  == INT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 16bit integer") {
			auto res = cast<uint16_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint16_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT16_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 32bit integer") {
			auto res = cast<int32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int32_t>);
			REQUIRE(res[0]  == INT32_MIN);
			REQUIRE(res[1]  == INT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 32bit integer") {
			auto res = cast<uint32_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint32_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT32_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to signed 64bit integer") {
			auto res = cast<int64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, int64_t>);
			REQUIRE(res[0]  == INT64_MIN);
			REQUIRE(res[1]  == INT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		WHEN("Casting to unsigned 64bit integer") {
			auto res = cast<uint64_t>(v);
			static_assert(std::same_as<decltype(res)::element_t, uint64_t>);
			REQUIRE(res[0]  == 0);
			REQUIRE(res[1]  == UINT64_MAX);
			REQUIRE(res[2]  == 3);
			REQUIRE(res[3]  == 4);
		}

		/*WHEN("Casting to float16 integer") {*/
		/*	auto res = cast<float16>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, float16>);*/
		/**/
		/*	REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
		/*}*/
		/**/
		/*WHEN("Casting to bfloat16 integer") {*/
		/*	auto res = cast<bfloat16>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, bfloat16>);*/
		/**/
		/*	REQUIRE_THAT(float(res[ 0]), Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 1]), Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 2]), Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
		/*	REQUIRE_THAT(float(res[ 3]), Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
		/*}*/
		/**/
		/*WHEN("Casting to float32 integer") {*/
		/*	auto res = cast<float>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, float>);*/
		/**/
		/*	REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(-INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel( INFINITY , eps<float>));*/
		/*	REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12f     , eps<float>));*/
		/*	REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12f     , eps<float>));*/
		/*}*/
		/**/
		/*WHEN("Casting to float64 integer") {*/
		/*	auto res = cast<double>(v);*/
		/*	static_assert(std::same_as<decltype(res)::element_t, double>);*/
		/**/
		/*	REQUIRE_THAT(res[ 0], Catch::Matchers::WithinRel(double(-INFINITY) , eps<double>));*/
		/*	REQUIRE_THAT(res[ 1], Catch::Matchers::WithinRel(double( INFINITY) , eps<double>));*/
		/*	REQUIRE_THAT(res[ 2], Catch::Matchers::WithinRel(3.12      , eps<double>));*/
		/*	REQUIRE_THAT(res[ 3], Catch::Matchers::WithinRel(4.12      , eps<double>));*/
		/*}*/
	}
}


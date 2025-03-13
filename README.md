# Unified-Intrinsics

## Goal

This library provides a unified interface for vector intrinsics across different architectures. It emulates intrinsics available on some platforms (e.g., ARM Neon) but missing on others (e.g., x86 or WebAssembly) using scalar operations or existing vector intrinsics.  This allows developers to write SIMD code once and have it run efficiently on a wider range of hardware.

## Features

*   **Abstraction:**  Provides a consistent API for vector operations, regardless of the underlying hardware.
*   **Emulation:**  Automatically emulates missing intrinsics for maximum portability.
*   **Performance:** Leverages native intrinsics when available and falls back to optimized emulation when necessary.
*   **Cross-Platform:** Supports ARM Neon, x86 (SSE, AVX256, AVX512), and WebAssembly SIMD.

## Supported Operations

The following categories of operations are currently supported:

*   Absolute Value
*   Addition
*   Bit Manipulation
*   Casting
*   Comparison
*   Division
*   Loading
*   Logical Operations
*   Vector Manipulation
*   Min/Max
*   Multiplication
*   Reciprocal
*   Rounding Operations
*   Shifting
*   Subtraction
*   Square Root
*   Shuffle
*   Matrix Support
*   `float16` and `bfloat16` Support

## Status

*   [x] Wrappers for `Emulated`, `ARM Neon`, `x86` (`SSE`, `AVX256`, and `AVX512`), and `Wasm SIMD`
    *   [x] Absolute Ops
    *   [x] Addition
    *   [x] Bit Manipulation
    *   [x] Casting
    *   [x] Comparison
    *   [x] Division
    *   [x] Loading
    *   [x] Logical Ops
    *   [x] Vector Manipulation
    *   [x] Min-Max
    *   [x] Multiplication
    *   [x] Reciprocal
    *   [x] Rounding Ops
    *   [x] Shifting
    *   [x] Subtraction
    *   [x] Square-root
    *   [x] Shuffle
    *   [x] Matrix support
    *   [x] Support for float16 and bfloat16
*   [x] Unit Tests
*   [x] CPU Information Retrieval using OS APIs
    *   [x] Cache and Instruction Cache Information
    *   [x] Memory Size
    *   [x] Cache Line Size
    *   [x] Compile-time Macro for Cache Line Size (Note: This macro provides an estimate.  For the most accurate value, use the `cpu_info` function at runtime and access the `cacheline` field.)
*   [ ] Examples (In Progress)

## TODO

*   Write comprehensive examples demonstrating library usage.

## Example
```cpp
#include "ui.hpp"

int main() {
    auto a = ui::native::f32(5); // [5, 5, 5, 5 ]
    auto b = ui::native::f32{ 1, 2, 3, 4 }; // or f32::load(1, 2, 3, 4);
    auto res = a * b;
    std::println("Result: {}", res); // Result: [5, 10, 15, 20]
    return 0;
}
```

## Class/Structs
### `Vec<N, T>`
```cpp
    template <std::size_t N, typename T>
    struct alignas(N * sizeof(T)) Vec {
        // Member variables
        base_type lo, hi;

        // Constructors
        constexpr Vec() noexcept = default;
        constexpr Vec(element_t x, element_t y) noexcept requires (N == 2);
        constexpr Vec(Vec<elements / 2, element_t> const& low, Vec<elements / 2, element_t> const& high) noexcept requires (elements != 4);
        constexpr Vec(element_t val) noexcept;
        constexpr Vec(Vec<2, element_t> xy, element_t z, element_t w) noexcept requires (elements == 4);
        constexpr Vec(Vec<2, element_t> xy, Vec<2, element_t> zw) noexcept requires (elements == 4);
        constexpr Vec(std::initializer_list<element_t> li) noexcept;
        constexpr Vec(std::span<element_t> li) noexcept;

        // Methods
        auto to_span() const noexcept -> std::span<element_t>;
        auto data() noexcept -> element_t*;
        auto data() const noexcept -> element_t const*;
        constexpr auto operator[](size_type k) noexcept -> element_t&;
        constexpr auto operator[](size_type k) const noexcept -> element_t;
        constexpr auto size() const noexcept;

        // Swizzling methods
        constexpr Vec<2, element_t>& xy() noexcept requires (N == 4);
        constexpr Vec<2, element_t>& zw() noexcept requires (N == 4);
        constexpr element_t& x() noexcept requires (N == 4 || N == 2);
        constexpr element_t& y() noexcept requires (N == 4 || N == 2);
        constexpr element_t& z() noexcept requires (N == 4);
        constexpr element_t& w() noexcept requires (N == 4);

        // Load and store methods
        static constexpr auto load(element_t const* const UI_RESTRICT in, size_type size) noexcept;
        constexpr auto store(element_t const* const UI_RESTRICT in, size_type size) noexcept;
    };
```

### `Vec<1, T>`
```cpp
template <typename T>
struct alignas(1 * sizeof(T)) Vec<1, T> {
    // Member variables
    element_t val;

    // Methods
    template <typename U>
    constexpr auto cast() const noexcept -> U;
    operator std::span<element_t const>() const noexcept;
    auto to_span() const noexcept;
    auto data() noexcept -> element_t*;
    auto data() const noexcept -> element_t const*;
    constexpr auto operator[](size_type k) noexcept -> element_t&;
    constexpr auto operator[](size_type k) const noexcept -> element_t;
    static constexpr auto load(T const* const UI_RESTRICT in, size_type size) noexcept;
    static constexpr auto load(std::span<T> data) noexcept;
    static constexpr auto load(element_t val) noexcept -> Vec;
    template <unsigned Lane, std::size_t M>
    static constexpr auto load(Vec<M, T> const&) noexcept -> Vec;
    constexpr auto store(T const* const UI_RESTRICT in, size_type size) noexcept;
};
```
### Example
```cpp
#include "ui.hpp"
int main() {
    auto v = ui::Vec<4, float>{1, 2, 3, 4};
    std::println("Vec: {}", v);
    return 0;
}
```

### `CpuInfo`
```cpp
struct CacheInfo {
    std::uint8_t level;
    unsigned size;
};

struct CpuInfo {
    std::vector<CacheInfo> cache;
    std::vector<CacheInfo> icache;
    unsigned cacheline;
    std::size_t mem;
};
```
#### Example
```cpp
#include "ui.hpp"

int main() {
    auto info = ui::cpu_info();
    std::println("CpuInfo: {}", info);
    return 0;
}
```

### `VecMat`

```cpp
template <std::size_t R, std::size_t C, typename T>
struct alignas(sizeof(Vec<C, T>)) VecMat {
    using vec_type = Vec<C, T>;
    using element_t = T;
    using size_type = std::size_t;

    static constexpr size_type rows = R;
    static constexpr size_type cols = C;

    vec_type val[R];

    // Constructors
    constexpr VecMat() noexcept = default;
    constexpr VecMat(VecMat const&) noexcept = default;
    constexpr VecMat(VecMat &&) noexcept = default;
    constexpr VecMat& operator=(VecMat const&) noexcept = default;
    constexpr VecMat& operator=(VecMat &&) noexcept = default;
    constexpr ~VecMat() noexcept = default;

    // Methods
    static auto load(element_t const* const in, size_type size) noexcept -> VecMat;
    static constexpr auto load(std::span<element_t> sp) noexcept -> VecMat;
    static constexpr auto load(element_t val) noexcept -> VecMat;

    template <is_vec... Ts>
    static constexpr auto load(Ts&&... args) noexcept -> VecMat;

    constexpr auto operator()(size_type r, size_type c) const noexcept -> element_t;
    constexpr auto operator()(size_type r, size_type c) noexcept -> element_t&;
    auto data() noexcept -> element_t*;
    auto data() const noexcept -> element_t const*;
    auto lo() const noexcept -> VecMat<R / 2, C, T>;
    auto hi() const noexcept -> VecMat<R / 2, C, T>;
};
```

### Overloaded Operators
#### 1. Logical Operators
```cpp
constexpr auto operator!(Vec<N, T> const& op) noexcept -> Vec<N, T>;
constexpr auto operator~(Vec<N, T> const& op) noexcept -> Vec<N, T>;
constexpr auto operator-(Vec<N, T> const& op) noexcept -> Vec<N, T>;
constexpr auto operator^(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator^(Vec<N, T> const& lhs, U const rhs) noexcept -> mask_t<N, T>;
constexpr auto operator^(U const lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator&(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator&(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator&(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator|(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator|(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator|(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator<<(Vec<N, T> const& lhs, Vec<N, std::make_unsigned_t<T>> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator<<(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator<<(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator>>(Vec<N, T> const& lhs, Vec<N, std::make_unsigned_t<T>> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator>>(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator>>(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
```

#### 2. Comparison Operators
```cpp
constexpr auto operator==(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator==(Vec<N, T> const& lhs, U const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator!=(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator!=(Vec<N, T> const& lhs, U const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator<=(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator<=(Vec<N, T> const& lhs, U const rhs) noexcept -> mask_t<N, T>;
constexpr auto operator<=(U const lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator<(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator<(Vec<N, T> const& lhs, U const rhs) noexcept -> mask_t<N, T>;
constexpr auto operator<(U const lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator>=(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator>=(Vec<N, T> const& lhs, U const rhs) noexcept -> mask_t<N, T>;
constexpr auto operator>=(U const lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator>(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
constexpr auto operator>(Vec<N, T> const& lhs, U const rhs) noexcept -> mask_t<N, T>;
constexpr auto operator>(U const lhs, Vec<N, T> const& rhs) noexcept -> mask_t<N, T>;
```

#### 3. Arithmetic Operators
```cpp
constexpr auto operator+(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator+(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator+(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator-(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator-(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator-(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator*(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator*(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator*(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator/(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator/(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator/(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator%(Vec<N, T> const& lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
constexpr auto operator%(Vec<N, T> const& lhs, U const rhs) noexcept -> Vec<N, T>;
constexpr auto operator%(U const lhs, Vec<N, T> const& rhs) noexcept -> Vec<N, T>;
```

## Type Alias
```cpp
using float2  = Vec< 2, float>;
using float4  = Vec< 4, float>;
using float8  = Vec< 8, float>;

using half2  = Vec< 2, float16>;
using half4  = Vec< 4, float16>;
using half8  = Vec< 8, float16>;

using bhalf2  = Vec< 2, bfloat16>;
using bhalf4  = Vec< 4, bfloat16>;
using bhalf8  = Vec< 8, bfloat16>;

using double2 = Vec< 2, double>;
using double4 = Vec< 4, double>;
using double8 = Vec< 8, double>;

using byte2   = Vec< 2,  std::uint8_t>;
using byte4   = Vec< 4,  std::uint8_t>;
using byte8   = Vec< 8,  std::uint8_t>;
using byte16  = Vec< 16, std::uint8_t>;

using int2    = Vec< 2, std::int32_t>;
using int4    = Vec< 4, std::int32_t>;
using int8    = Vec< 8, std::int32_t>;

using ushort2 = Vec< 2, std::uint16_t>;
using ushort4 = Vec< 4, std::uint16_t>;
using ushort8 = Vec< 8, std::uint16_t>;

using uint2   = Vec< 2, std::uint32_t>;
using uint4   = Vec< 4, std::uint32_t>;
using uint8   = Vec< 8, std::uint32_t>;

using long2   = Vec< 2, std::int64_t>;
using long4   = Vec< 4, std::int64_t>;
```

### Example
```cpp
#include "ui.hpp"
int main() {
    auto v = ui::float4{1, 2, 3, 4};
    std::println("Vec: {}", v);
    return 0;
}
```

## Native Type Alias
```cpp
using f16  = Vec< 8 * NativeSizeFactor, float16>;
using bf16 = Vec< 8 * NativeSizeFactor, bfloat16>;
using f32  = Vec< 4 * NativeSizeFactor, float>;
using f64  = Vec< 2 * NativeSizeFactor, double>;
using u8   = Vec<16 * NativeSizeFactor, std::uint8_t>;
using u16  = Vec< 8 * NativeSizeFactor, std::uint16_t>;
using u32  = Vec< 4 * NativeSizeFactor, std::uint32_t>;
using u64  = Vec< 2 * NativeSizeFactor, std::uint64_t>;
using i8   = Vec<16 * NativeSizeFactor, std::int8_t>;
using i16  = Vec< 8 * NativeSizeFactor, std::int16_t>;
using i32  = Vec< 4 * NativeSizeFactor, std::int32_t>;
using i64  = Vec< 2 * NativeSizeFactor, std::int64_t>;
```

### Example
```cpp
#include "ui.hpp"
int main() {
    auto v = ui::native::f32{1, 2, 3, 4};
    std::println("Vec: {}", v);
    return 0;
}
```

## Function Reference

### Absolute Operations

#### 1. `abs_diff`
```cpp
abs_diff(Vec lhs, Vec rhs) -> Vec
```
##### Description
Computes `lhs - rhs` if `lhs` is bigger than `rhs`; otherwise, `rhs - lhs`.

#### 2. `widening_abs_diff`
```cpp
widening_abs_diff(Vec lhs, Vec rhs) -> Vec
```
##### Description
Computes `abs_diff` but widens the vector type like 8bit integer to 16bit integer.

#### 3. `abs_acc_diff`
```cpp
abs_acc_diff(Vec acc, Vec lhs, Vec rhs) -> Vec
```
##### Description
Computes `acc + acc_diff(lhs, rhs)`

#### 4. `abs`
```cpp
abs(Vec v) -> Vec
```
##### Description
It takes an absolute value of each vector elements. `|v|`.

#### 4. `sat_abs`
```cpp
sat_abs(Vec v) -> Vec
```
##### Description
It takes absolute value while saturating like 8bit integer `-128` will be turned to `127`; otherwise, normal `abs` returns `-128` since `128` cannot be represented in signed 8bit integer.


### Addition

#### 1. `add`
```cpp
add(Vec lhs, Vec rhs) -> Vec
```
##### Description
Computes `lhs + rhs`.

#### 2. `widening_add`
```cpp
widening_add(Vec lhs, Vec rhs) -> Vec
```
##### Description
Computes `lhs + rhs` but widens the return type like 8bit integer to 16bit integer.

#### 3. `halving_add`
```cpp
halving_add(Vec lhs, Vec rhs) -> Vec
```
##### Description
Computes `(lhs + rhs)/2` and rounds if needed

#### 4. `high_narrowing_add`
```cpp
high_narrowing_add(Vec lhs, Vec rhs) -> Vec
```
##### Description
It Adds the left and right hand side but returns the upper half of bits. So, it'll narrow the result type like 16bit integer to 8bit integer. Ex: let `0b1110'0101'1111'0000` is the result after adding two 16bit integer then result will be `0b1110'0101`.

#### 5. `sat_add`
```cpp
sat_add(Vec lhs, Vec rhs) -> Vec
```
##### Description
It adds the lhs and rhs while saturating the result. Ex: `128 + anything` in 8bit integer will always return `128`.

#### 6. `padd`
```cpp
padd(Vec lhs, Vec rhs) -> Vec
```
##### Description
It adds next value in each register and combine both lhs and rhs together.
```
a: [1, 2, 3, 4]
b: [5, 6, 7, 8]
padd(a,b): [1 + 2, 3 + 4, 5 + 6, 7 + 8] => [3, 7, 11, 15]
```

#### 7. `fold`
```cpp
fold(Vec v, op::padd_t) -> Value
//          OR
fold(Vec v, op::add_t) -> Value
```

##### Description
It reduces the vector using addition and returns the scalar value.

```
a: [1, 2, 3, 4]
fold(a, op::add_t{}): 1 + 2 + 3 + 4 => 10
```

#### 6. `widening_padd`
```cpp
widening_padd(Vec v) -> Vec
```
##### Description
It is similar to `padd` but result type is promoted to next big integer and reduces the resulting vector register to half.

```
a: [1, 2, 3, 4]
widening_padd(a): [1 + 2, 3 + 4] => [3, 7]
```
#### 7. `widening_padd`
```cpp
widening_padd(Vec<N / 2, W> a, Vec<N, T> v) -> Vec<N / 2, W> where W > T
```
##### Description
It adds the `a` and adjacent pair together.

```
a: [1, 2]
b: [5, 6, 7, 8]
widening_padd(a, b): [1 + 5 + 6, 2 + 7 + 8] => [12, 17]
```
#### 8. `widening_fold`
```cpp
widening_fold(Vec v, op::add_t) -> WidenedValue
```

##### Description
It is similar to addition fold but the result type is widened to avoid truncation.

### Bits

#### 1. `count_leading_sign_bits`
```cpp
count_leading_sign_bits(Vec v) -> Vec
```
##### Description
Counts the number of signed bits (0/1).

```
a: i8 = [-1, 2]
count_leading_sign_bits(a): [7, 5]
```

#### 2. `count_leading_zeros`
```cpp
count_leading_zeros(Vec v) -> Vec
```
##### Description
Counts leading zeros in each lane.

#### 3. `popcount`
```cpp
popcount(Vec v) -> Vec
```
##### Description
Counts the number of '1s' inside a bit pattern.

#### 4. `bitwise_clear`
```cpp
bitwise_clear(Vec a, Vec b) -> Vec
```
##### Description
Computes `a & ~b`.

#### 5. `bitwise_select` or `if_then_else`
```cpp
bitwise_select(Mask condition, Vec true, Vec false) -> Vec;
if_then_else(Mask condition, Vec true, Vec false) -> Vec;
```
##### Description
If condition is true then it selects an element from true lane otherwise false lane.

```
a: i8 = [   1, 2,    3, 4]
b: i8 = [   5, 3,    1, 0]
c: i8 = [0xff, 0, 0xff, 0]
bitwise_select(c, a, b): [1, 3, 3, 0]
```

### Casting

#### 1. `cast`
```cpp
cast<To>(Vec<N, From> v) -> Vec<N, To>
```
##### Description
Convert one type to another type. It could be downcast to type promotion.

```
a: f32 = [1.2, 0.33, 123, 2.5]
cast<int>(a): [1, 0, 123, 2]
```
#### 2. `sat_cast`
```cpp
sat_cast<To>(Vec<N, From> v) -> Vec<N, To>
```
##### Description
It's similar to `cast` but it saturates the result

```
a: i16 = [2000, -1200, 12, 0]
sat_cast<int8_t>(a): [127, -128, 12, 0]
```

#### 2. `rcast`
```cpp
rcast<To>(Vec<N, From> v) -> Vec<N, To>
```
##### Description
It acts like C++'s `reinterpret_cast`. It interprets underlying data to another type.

### Division

#### 1. `div`
```cpp
div(Vec num, Vec den) -> Vec
```
##### Description
As the name suggests, it divides the numbers.
> **_NOTE:_** For integer division, it casts the number to floating-point and then divides them on some architecture since they do not support SIMD division.


#### 2. `div`
```cpp
rem(Vec num, Vec den) -> Vec
```
> **_NOTE:_** There is no native support for remainder in most of the architecture we support so we have to use division and subtraction with rounding.


### Load

#### 1. `load<N>`
```cpp
load<N>(T val) -> Vec<N, T>
```
##### Description
Broadcasts the scalar value across the lanes and returns a new vector.
```cpp
auto v = load<8>(2); // [2, 2, 2, 2, 2, 2, 2, 2]
```
#### 2. `load<N, Lane>`
```cpp
load<N, Lane>(Vec<M, T> v) -> Vec<N, T> where Lane < M
```
##### Description
Broadcasts the lane value from given vector.
```cpp
auto v = ui::float4 {1, 2, 3, 4};
auto a = ui::load<4, 0>(v); // [1, 1, 1, 1]
```
#### 2. `strided_load`
```cpp
strided_load(T const* data, Vec<N, T> a, Vec<N, T> b) -> void;
strided_load(T const* data, Vec<N, T> a, Vec<N, T> b, Vec<N, T> c) -> void;
strided_load(T const* data, Vec<N, T> a, Vec<N, T> b, Vec<N, T> c, Vec<N, T> d) -> void;
```
##### Description
It has three overloads. The number `Vec` arguments tell you the stride it'll take. For example: two `Vec` has stride of 2, three `Vec` has stride 3 and the last one has stride 4.

```
data = [1, 2, 3, 4, 5, 6, 7, 8, .... 100]
a: i8 = []
b: i8 = []
strided_load(data, a, b)
a => [1, 3, 5, 7, ...]
b => [2, 4, 6, 8, ...]
```
```
data = [1, 2, 3, 4, 5, 6, 7, 8, .... 100]
a: i8 = []
b: i8 = []
c: i8 = []
strided_load(data, a, b, c)
a => [1, 4, 7, 10, ...]
b => [2, 5, 8, 11, ...]
c => [3, 6, 9, 12, ...]
```

### Logical

#### 1. `negate`
```cpp
negate(Vec v) -> Vec
```
##### Description
It negates each elements inside the vector/simd register. `-v`.

```
data = [1, 2, -3, 4]
negate(data): [-1, -2, 3, -4]
```

#### 2. `sat_negate`
```cpp
sat_negate(Vec v) -> Vec
```
##### Description
It negates each elements inside the vector/simd register while saturating. `-v`.

```
data: i8 = [1, 2, -128, 4]
negate(data): [-1, -2, 127, -4]
```

#### 3. `bitwise_not`
```cpp
bitwise_not(Vec v) -> Vec
```
##### Description
It is equivalent to C++'s `~` operator.

#### 4. `bitwise_and`
```cpp
bitwise_and(Vec lhs, Vec rhs) -> Vec
```
##### Description
It is equivalent to C++'s `&` operator.

#### 5. `bitwise_or`
```cpp
bitwise_or(Vec lhs, Vec rhs) -> Vec
```
##### Description
It is equivalent to C++'s `|` operator.

#### 6. `bitwise_or`
```cpp
bitwise_xor(Vec lhs, Vec rhs) -> Vec
```
##### Description
It is equivalent to C++'s `^` operator.

#### 7. `bitwise_ornot`
```cpp
bitwise_xor(Vec lhs, Vec rhs) -> Vec
```
##### Description
It is equivalent to `lhs | ~rhs`.

#### 8. `bitwise_notand`
```cpp
bitwise_xor(Vec lhs, Vec rhs) -> Vec
```
##### Description
It is equivalent to `~lhs & rhs`.

### Vector Manipulation

#### 1. `copy`
```cpp
copy<ToLane, FromLane>(Vec to, Vec from) -> Vec
```
##### Description
It allows to copy a element `from` vector at a given position to the `to` at a given position.

```
to = [1, 2, 3, 4]
from = [5, 6, 7, 8]
negate<0, 1>(to, from): [6, 2, 3, 4]
```

#### 2. `reverse_bits`
```cpp
reverse_bits(Vec v) -> Vec
```
##### Description
It reverses the bit pattern of each elements. Ex: `0b1111'0000` will be converted to `0b0000'1111`


#### 3. `reverse`
```cpp
reverse(Vec v) -> Vec
```
##### Description
It reverse the elements of the vector register. Ex: `[1, 2, 3, 4]` will be turned to `[4, 3, 2, 1]`.

#### 4. `zip_low`
```cpp
zip_low(Vec a, Vec b) -> Vec
```
##### Description
It zips the lower part of the vector together.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
zip_low(a, b): [1, 5, 2, 6]
```
#### 5. `zip_high`
```cpp
zip_high(Vec a, Vec b) -> Vec
```
##### Description
It zips the upper part of the vector together.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
zip_high(a, b): [3, 7, 4, 8]
```
#### 6. `unzip_low`
```cpp
unzip_low(Vec a, Vec b) -> Vec
```
##### Description
It puts odd part together from the first and second vector register and combine them.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
unzip_low(a, b): [1, 3, 5, 7]
```
#### 7. `unzip_high`
```cpp
unzip_high(Vec a, Vec b) -> Vec
```
##### Description
It puts even part together from the first and second vector register and combine them.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
unzip_high(a, b): [2, 4, 6, 8]
```
#### 8. `transpose_low`
```cpp
transpose_low(Vec a, Vec b) -> Vec
```
##### Description
It is similar to zip but works with odd position.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
transpose_low(a, b): [1, 5, 3, 7]
```

#### 9. `transpose_high`
```cpp
transpose_high(Vec a, Vec b) -> Vec
```
##### Description
It is similar to zip but works with even position.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
transpose_high(a, b): [2, 6, 4, 8]
```

### Matrix Operations

#### 1. `transpose`

```cpp
transpose(VecMat v) -> VecMat;
```
##### Description
Transposes the matrix.

#### 2. `mul`

```cpp
mul(VecMat<N, M, T> c, VecMat<N, K, T> a, VecMat<K, M, T> b) -> VecMat;
mul(VecMat<N, K, T> a, VecMat<K, M, T> b) -> VecMat;
```
##### Description
Multiplies two matrices and adds it the `c`. `c + a * b`.
> **_NOTE:_** This implementation uses broadcasting an element and assumes the matrix is transposed.

#### 3. `mul2`

```cpp
mul2(VecMat<N, M, T> c, VecMat<N, K, T> a, VecMat<K, M, T> b) -> VecMat;
mul2(VecMat<N, K, T> a, VecMat<K, M, T> b) -> VecMat;
```
##### Description
Multiplies two matrices and adds it the `c`. `c + a * b`. 
> **_NOTE:_** This implementation transposes the matrix and uses dot product.

### Min-Max

#### 1. `max`
```cpp
max(Vec a, Vec b) -> Vec
```
##### Description
It returns maximum in each position.

```
a = [1, 2, 3, 4]
b = [2, 2, 2, 2]
max(a, b): [2, 2, 3, 4]
```

#### 2. `min`
```cpp
min(Vec a, Vec b) -> Vec
```
##### Description
It returns minimum in each position.

```
a = [1, 2, 3, 4]
b = [2, 2, 2, 2]
max(a, b): [1, 2, 2, 2]
```

#### 3. `maxnm`
```cpp
maxnm(Vec a, Vec b) -> Vec
```
##### Description
It returns number-maximum in each position, which is similar to maximum but treats `NaN` differently. It returns `max(NaN, 10) = 10`, `max(10, NaN) = 10`, or `max(NaN, NaN) = NaN`

#### 4. `minnm`
```cpp
minnm(Vec a, Vec b) -> Vec
```
##### Description
It returns number-minimum in each position, which is similar to minimum but treats `NaN` differently. It returns `min(NaN, 10) = 10`, `min(10, NaN) = 10`, or `min(NaN, NaN) = NaN`

#### 5. `pmax`
```cpp
pmax(Vec a, Vec b) -> Vec
```
##### Description
It takes adjacent maximum.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
pmax(a, b): [max(1, 2), max(3, 4), max(5, 6), max(7, 8)]
```

#### 6. `pmaxnm`
```cpp
pmaxnm(Vec a, Vec b) -> Vec
```
##### Description
It's a mixture of `pmax` and `maxnm`.

#### 7. `pmin`
```cpp
pmin(Vec a, Vec b) -> Vec
```
##### Description
It takes adjacent minimum.

```
a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
pmin(a, b): [min(1, 2), min(3, 4), min(5, 6), min(7, 8)]
```

#### 8. `pminnm`
```cpp
pminnm(Vec a, Vec b) -> Vec
```
##### Description
It's a mixture of `pmin` and `minnm`.

#### 9. `fold`
```cpp
fold(Vec a, op::pmax_t) -> Value;
fold(Vec a, op::pmaxnm_t) -> Value;
fold(Vec a, op::max_t) -> Value;
fold(Vec a, op::maxnm_t) -> Value;
```
##### Description
It reduces the vector based on the `tag`. `pmax_t` and `max_t` uses difference intrinsics on `arm` but are same on different platform. 

#### 10. `fold`
```cpp
fold(Vec a, op::pmin_t) -> Value;
fold(Vec a, op::pminnm_t) -> Value;
fold(Vec a, op::min_t) -> Value;
fold(Vec a, op::minnm_t) -> Value;
```
##### Description
It reduces the vector based on the `tag`. `pmin_t` and `min_t` uses difference intrinsics on `arm` but are same on different platform. 

### Multiplication Operation

#### 1. `mul`
```cpp
mul(Vec lhs, Vec rhs) -> Vec;
mul(Vec<N, T> v, T constant) -> Vec;
mul(Vec<N, T> v, T constant) -> Vec;
mul<Lane>(Vec<N, T> a, Vec<M, T> v, op::add_t) -> Vec<N, T> where Lane < M;
```
##### Description
It multiplies two vectors.

#### 2. `mul_acc`
```cpp
mul_acc(Vec acc, Vec lhs, Vec rhs, op::add_t) -> Vec;
mul_acc(Vec acc, Vec lhs, Vec rhs, op::sub_t) -> Vec;
mul_acc(Vec<N, W> acc, Vec<N, T> lhs, Vec<N, T> rhs, op::add_t) -> Vec<N, W> where W > T;
mul_acc(Vec<N, W> acc, Vec<N, T> lhs, Vec<N, T> rhs, op::sub_t) -> Vec<N, W> where W > T;
mul_acc(Vec<N, T> acc, Vec<N, T> a, T constant, op::add_t) -> Vec;
mul_acc(Vec<N, T> acc, Vec<N, T> a, T constant, op::sub_t) -> Vec;
```
##### Description
It multiplies two vectors and add it to or subtract it from the accumulator. `acc +/- (lhs * rhs)`

#### 3. `fused_mul_acc`
```cpp
fused_mul_acc(Vec acc, Vec lhs, Vec rhs, op::add_t) -> Vec;
fused_mul_acc(Vec acc, Vec lhs, Vec rhs, op::sub_t) -> Vec;
```
##### Description
It does the same as `mul_acc` but uses fused intrinsic.

> **_NOTE:_** Platforms that does not support fused multiplication instruction, it'll fallback to two `mul_acc`

#### 4. `fused_mul_acc` / `mul_acc`
```cpp
fused_mul_acc<Lane>(Vec<N, T> acc, Vec<N, T> a, Vec<M, T> v, op::add_t) -> Vec where Lane < M;
fused_mul_acc<Lane>(Vec<N, T> acc, Vec<N, T> a, Vec<M, T> v, op::sub_t) -> Vec where Lane < M;
mul_acc<Lane>(Vec<N, T> acc, Vec<N, T> a, Vec<M, T> v, op::add_t) -> Vec where Lane < M;
mul_acc<Lane>(Vec<N, T> acc, Vec<N, T> a, Vec<M, T> v, op::sub_t) -> Vec where Lane < M;
```
##### Description
It takes the element from the vector `v` at a given position and multiplies it with `a`. `acc +/- a * v[Lane]`

#### 5. `widening_mul`
```cpp
widening_mul(Vec<N, T> lhs, Vec<N, T> rhs) -> Vec<N, W> where W > T;
widening_mul<Lane>(Vec<N, T> v, Vec<M, T> rhs) -> Vec<N, W> where W > T && Lane < M;
widening_mul(Vec<N, T> a, T constant) -> Vec<N, W> where W > T;
```
##### Description
It is similar to `mul` with it widens the resultant vectors.

#### 6. `mul_acc`
```cpp
mul_acc(Vec<N, W> acc, Vec<N, T> a, T constant, op::add_t) -> Vec where W > T;
mul_acc(Vec<N, W> acc, Vec<N, T> a, T constant, op::sub_t) -> Vec where W > T;
```
##### Description
It is similar to `mul_acc` with it widens the resultant vectors.

#### 7. `fused_mul_acc`
```cpp
fused_mul_acc(Vec<N, W> acc, Vec<N, T> a, T constant, op::add_t) -> Vec where W > T;
fused_mul_acc(Vec<N, W> acc, Vec<N, T> a, T constant, op::sub_t) -> Vec where W > T;
```
##### Description
It is similar to `fused_mul_acc` with it widens the resultant vectors.

### Shuffle/Permute

#### 1. `shuffle`
```cpp
shuffle<Is...>(Vec<N, T> v) -> Vec<count(Is...), T> where count(Is...) is power of 2;
```
##### Description
It shuffle the element around based on the indices.

> **_NOTE:_** It's a compiler dependent function. If compiler is GCC or Clang, it'll use builtin to shuffle around the vector, but other compilers it leaves the optimization to the compiler.

```
a = [1, 2, 3, 4]
shuffl<3, 2>(a) => [4, 3]
```

### Prefetch

```cpp
enum class PrefetchRW { Read, Write, RW };
enum class PrefetchLocality { None, Low, Medium, High };

prefetch<RW = PrefetchRW::Read, Locality = PrefetchLocality::High>(T const* const data) -> void
```
##### Description
It a no-op for platform that does not support it, but for `x86` and `ARM`, it'll generate prefetch instruction.

### Reciprocal Operations

#### 1. `reciprocal_estimate`
```cpp
reciprocal_estimate(Vec v) -> Vec
```
##### Description
It estimates `1/v`.

#### 2. `reciprocal_refine`
```cpp
reciprocal_refine(Vec v, Vec previous_estimate) -> Vec
```
##### Description
It refines the previously generated estimate by one step.

```
a: f32 = [1, 2, 3, 4]
e = reciprocal_estimate(a) => [0.9980469, 0.49902344, 0.3330078, 0.24951172]
reciprocal_refine(a, e) => [1.0019531, 1.0019531, 1.0009766, 1.0019531] 
```

#### 3. `sqrt_inv_estimate`
```cpp
sqrt_inv_estimate(Vec v) -> Vec
```
##### Description
It estimates `1/sqrt(v)`.

#### 4. `sqrt_inv_refine`
```cpp
sqrt_inv_refine(Vec v, Vec previous_estimate) -> Vec
```
##### Description
It refines the previously generated estimate by one step.

#### 5. `exponent_reciprocal_estimate`
```cpp
exponent_reciprocal_estimate(Vec v) -> Vec
```
##### Description
It estimates `1/float_exponent(v)`. Ex: `1.101 * 2^x` => `1/(2^x)` => `2^-x`

### Rounding
```cpp
round<Mode = std::float_round_style::round_to_nearest>(Vec v) -> Vec
```
##### Description
It rounds a floating-point based on the mode.

### Shift

#### 1. `shift_left`
```cpp
shift_left(Vec<N, T> v, Vec<N, Unsigned(T)> count) -> Vec<N, T>;
```
##### Description
It maps to `v << count` in C++.

#### 2. `shift_left`
```cpp
shift_left<Count>(Vec<N, T> v) -> Vec<N, T>;
```
##### Description
It maps to `v << Count` in C++.

#### 3. `sat_shift_left`
```cpp
sat_shift_left(Vec<N, T> v, Vec<N, Unsigned(T)> count) -> Vec<N, T>;
```
##### Description
It similar to `shift_left` while saturating the result.

#### 4. `sat_shift_left`
```cpp
shift_left<Count>(Vec<N, T> v) -> Vec<N, T>;
```
##### Description
It similar to `shift_left` while saturating the result.

#### 5. `rounding_shift_left`
```cpp
rounding_shift_left(Vec<N, T> v, Vec<N, Unsigned(T)> count) -> Vec<N, T>;
```
##### Description
It similar to `shift_left` while rounding the result.

#### 6. `sat_rounding_shift_left`
```cpp
sat_rounding_shift_left(Vec<N, T> v, Vec<N, Unsigned(T)> count) -> Vec<N, T>;
```
##### Description
It similar to `shift_left` while rounding and saturating the result.

#### 7. `widening_shift_left`
```cpp
widening_shift_left<Count>(Vec<N, T> v) -> Vec<N, W> where W > T;
```
##### Description
It similar to `shift_left` and the result is widens.

#### 8. `insert_shift_left`
```cpp
insert_shift_left<Count>(Vec a, Vec b) -> Vec;
```
##### Description
It shifts the vector `b` to left by count and inserts the bits in the empty space left by shifting. 
Operation: let mask be `(1 << Count) - 1` then `(a & mask) | ((b << Count) & ~mask)`.

#### 9. `shift_right`
```cpp
shift_right(Vec<N, T> v, Vec<N, Unsigned(T)> count) -> Vec<N, T>;
```
##### Description
It maps to `v >> count` in C++.

#### 10. `shift_right`
```cpp
shift_right<Count>(Vec v) -> Vec;
```
##### Description
It maps to `v >> count` in C++.

#### 11. `sat_shift_right`
```cpp
sat_shift_right(Vec<N, T> v, Vec<N, Unsigned(T)> count) -> Vec<N, T>;
```
##### Description
It does the same as right shift while saturating.

#### 12. `rounding_shift_right`
```cpp
rounding_shift_right<Count>(Vec v) -> Vec;
```
##### Description
It does the same as right shift while rounding.

#### 13. `sat_rounding_shift_right`
```cpp
rounding_shift_right<Count>(Vec v) -> Vec;
```
##### Description
It does the same as right shift while rounding and saturating.

#### 14. `rounding_shift_right`
```cpp
rounding_shift_right_accumulate<Count>(Vec acc, Vec v) -> Vec;
```
##### Description
It does the same as right shift while adding it to `acc`. `acc + (v << Shift)`

#### 15. `narrowing_shift_right`
```cpp
narrowing_shift_right<Count>(Vec<N, T> v) -> Vec<N, NT> where T > NT; 
```
##### Description
It does the same as right shift and narrows the result.

#### 16. `rounding_narrowing_shift_right`
```cpp
narrowing_shift_right<Count>(Vec<N, T> v) -> Vec<N, NT> where T > NT; 
```
##### Description
It does the same as right shift and narrows the result while rounding.

#### 17. `sat_narrowing_shift_right`
```cpp
sat_narrowing_shift_right<Count>(Vec<N, T> v) -> Vec<N, NT> where T > NT; 
```
##### Description
It does the same as right shift and narrows the result while saturating.

#### 18. `sat_rounding_narrowing_shift_right`
```cpp
sat_rounding_narrowing_shift_right<Count>(Vec<N, T> v) -> Vec<N, NT> where T > NT; 
```
##### Description
It does the same as right shift and narrows the result while saturating and rounding.

#### 19. `sat_unsigned_narrowing_shift_right`
```cpp
sat_unsigned_narrowing_shift_right<Count>(Vec<N, T> v) -> Vec<N, UnsignedNarrowedType> where T > UnsignedNarrowedType;
```
##### Description
It does the same as right shift and narrows the result and converts the result to unsigned while saturating.

#### 20. `sat_rounding_unsigned_narrowing_shift_right`
```cpp
sat_rounding_unsigned_narrowing_shift_right<Count>(Vec<N, T> v) -> Vec<N, UnsignedNarrowedType> where T > UnsignedNarrowedType;
```
##### Description
It does the same as right shift and narrows the result and converts the result to unsigned while saturating and rounding.

#### 21. `insert_shift_left`
```cpp
insert_shift_right<Count>(Vec a, Vec b) -> Vec;
```
##### Description
It shifts the vector `b` to right by count and inserts the bits in the empty space left by shifting. 
Operation: let mask be `~UT(0) >> Count` then `(a & ~mask) | (b >> Count)`.

### Square-Root

```cpp
sqrt(Vec v) -> Vec;
```
##### Description
It applies `sqrt` function each elements of the vector.

### Subtraction

#### 1. `sub`

```cpp
sub(Vec lhs, Vec rhs) -> Vec;
```
##### Description
It maps to `lhs - rhs` in C++.

#### 2. `widening_sub`

```cpp
widening_sub(Vec<N, T> lhs, Vec<N, T> rhs) -> Vec<N, W> where W > T;
```
##### Description
It subtracts the operands while widening the result.

#### 3. `halving_sub`

```cpp
halving_sub(Vec lhs, Vec rhs) -> Vec
```
##### Description
Computes `(lhs - rhs) / 2`

#### 4. `high_narrowing_sub`

```cpp
high_narrowing_sub(Vec lhs, Vec rhs) -> Vec
```
##### Description
It computes `lhs - rhs` and returns upper part of the bit pattern.

#### 5. `sub`

```cpp
sat_sub(Vec lhs, Vec rhs) -> Vec;
```
##### Description
It subtracts operand while saturating.

### Utilities

#### 1. `any`

```cpp
any(Vec v) -> bool;
```
##### Description
Returns true if any of the elements is non-zero otherwise false.

#### 2. `all`

```cpp
all(Vec v) -> bool;
```
##### Description
Returns true if all the elements are non-zero otherwise false.

#### 3. `ceil`

```cpp
ceil(Vec v) -> Vec;
```
##### Description
Equivalent to `std::ceil`

#### 4. `floor`

```cpp
floor(Vec v) -> Vec;
```
##### Description
Equivalent to `std::floor`

#### 5. `trunc`

```cpp
trunc(Vec v) -> Vec;
```
##### Description
Equivalent to `std::trunc`

#### 6. `frac`

```cpp
frac(Vec v) -> Vec;
```
##### Description
Computes the fractional part of the number. `v - floor(v)`.

#### 7. `div255`

```cpp
div255(Vec<N, u16> v) -> Vec<N, u8>;
```
##### Description
Transforms a number to in 16bit integer to 8bit integer.

#### 8. `approx_scale`

```cpp
approx_scale(Vec<N, u8> x, Vec<N, u8> y) -> Vec<N, u8>;
```
##### Description
It computes `(x * y + x) / 256`

#### 9. `dot`

```cpp
dot(Vec<N, u8> x, Vec<N, u8> y) -> Vec<N, u8>;
```
##### Description
It computes dot product of the given vectors.

#### 10. `cross`

```cpp
cross(Vec<2, T> x, Vec<2, T> y) -> T;
```
##### Description
It computes cross product of vectors of length 2.

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

#### 5. `bitwise_select`
```cpp
bitwise_select(Mask condition, Vec true, Vec false) -> Vec
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

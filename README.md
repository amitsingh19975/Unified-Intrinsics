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



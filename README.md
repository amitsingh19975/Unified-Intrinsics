# Unified-Intrinsics

## Goal
This library aims to unify vector intrinsics. If an instrinsic exists in arm but does not exist in x86, it will be emulated using scaler or previously defined vector instrinsics.

## TODO
- [x] Write wrappers for `ARM Neon`
    - [x] Absolute Ops
    - [x] Addition
    - [x] Bit Manipulation
    - [x] Casting
    - [x] Comparision
    - [x] Division
    - [x] Loading
    - [x] Logical Ops
    - [x] Vector Manipulation
    - [x] Min-Max
    - [x] Multiplication
    - [x] Reciprocal
    - [x] Rounding Ops
    - [x] Shifting
    - [x] Subtraction
    - [x] Square-root
    - [x] Shuffle
    - [x] Matrix support
    - [x] Support for float16 and bfloat16
- [x] Generic implementation (Let the compiler decide)
- [x] Write wrappers `x86` (`SSE`, `AVX 256`, and `AVX512`)
    - [x] Absolute Ops
    - [x] Addition
    - [x] Bit Manipulation
    - [x] Casting
    - [x] Comparision
    - [x] Division
    - [x] Loading
    - [x] Logical Ops
    - [x] Vector Manipulation
    - [x] Min-Max
    - [x] Multiplication
    - [x] Reciprocal
    - [x] Rounding Ops
    - [x] Shifting
    - [x] Subtraction
    - [x] Square-root
    - [x] Shuffle
    - [x] Matrix support
    - [x] Support for float16 and bfloat16
- [x] Writing tests
- [x] CpuInfo using OS APIs
    - [x] Get cache info and instruction cache info
    - [x] Get memory size
    - [x] Get cacheline size
    - [x] Add compile-time macro for cacheline (it is not accurate; for accurate size, use `cpu_info` at runtime and read the field `cacheline`)
- [ ] Writing proper examples


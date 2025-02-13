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
- [ ] Write wrappers `x86` (`SSE`, `AVX`, `AVX2`, and `AVX512`)
- [ ] Writing proper examples
- [ ] Writing tests


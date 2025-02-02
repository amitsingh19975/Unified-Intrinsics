# Unified-Intrinsics

## Goal
This library aims to unify vector intrinsics. If an instrinsic exists in arm but does not exist in x86, it will be emulated using scaler or previously defined vector instrinsics.

## TODO
- [ ] Write wrappers for `ARM Neon`
    - [x] Addition
    - [x] Subtraction
    - [x] Multiplication
    - [x] Division
    - [x] Min-Max
    - [x] Absolute Ops
    - [x] Rounding Ops
    - [ ]  Reciprocal
- [ ] Write wrappers `x86` (`SSE`, `AVX`, `AVX2`, and `AVX512`)
- [ ] Write a compatability layer for both layers
- [ ] Try to emulate float16 if it does not exist 
- [ ] Try to emulate bfloat if it does not exist 


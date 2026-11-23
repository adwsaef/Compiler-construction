# Compiler construction

Compile to x86-64 with `uv run compiler/compile.py program.lq program.s`, or run with `uv run compiler/compile.py program.lq --interp`.

## Type-checking conventions

- Type checking runs before `shrink`. Use `--no-type-check` for unannotated programs. Both modes check function names and argument counts.
- Parameters and return types require bare `int` or `bool` annotations. One annotation per variable suffices for the whole function, even inside a branch or after the first assignment. The compiler does not check whether a variable has been initialized before use.
- Function signatures are collected up front, so mutual recursion works. Function names are reserved for direct calls.
- `int` and `bool` are distinct. Write `if n != 0:` for an integer condition. The checker rejects `True + 1` and `0 == False`.
- Falling off the end of a function returns zero (`False` for a Boolean result). Explicit returns must match the declared type.

Use `--interp-after type_check` to run the source or `--interp-after PASS` to run an intermediate form.

When both multiplication operands are integer constants, the compiler rejects products outside the signed 64-bit range. Runtime multiplication wraps on overflow.

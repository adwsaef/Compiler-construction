This directory contains support files for the compiler projects

To use this in your uv project, add to `pyproject.toml`

```
dependencies = [
    "mrj-support",
]

[tool.uv.sources]
mrj-support = { path = "../support", editable = true }
```

(you may need adjust `../support` to where support dir lives relative to your project)

- `compiler_skel.py` - skeleton of the compiler class
- `ast_utils` - additions and extensions to Python AST
- `x86_ast.py` - x86 assembly AST
- `runtime.c` - runtime to link with generated assembly files

"""Function definitions, in the two shapes the middle end needs.

`Lfun` and `Lfun^mon` carry a definition whose body is a list of statements;
`CFun`, after explicate_control, carries one whose body is a dictionary of
labelled basic blocks.  They are separate classes rather than one reused
`ast.FunctionDef` so that `body` means exactly one thing in each pass.

`Begin` and `Goto` still come from cif.py -- explicate_control's output
language is unchanged apart from being per function.
"""
from ast import stmt
from dataclasses import dataclass, field

from ast_utils import dedent, indent
from cif import BasicBlocks


@dataclass
class FunDef:
    """A definition in Lfun / Lfun^mon: a body of statements."""
    name: str
    params: list[str]
    body: list[stmt]
    __match_args__ = ("name", "params", "body")

    def __str__(self):
        header = f"def {self.name}({', '.join(self.params)}):\n"
        indent()
        body = ''.join(str(s) for s in self.body)
        dedent()
        return header + body


@dataclass
class FunProgram:
    """A whole program: nothing but function definitions, one named main."""
    defs: list[FunDef]
    __match_args__ = ("defs",)

    def __str__(self):
        return '\n'.join(str(d) for d in self.defs)


@dataclass
class CFunDef:
    """A definition in CFun: a body of labelled basic blocks."""
    name: str
    params: list[str]
    blocks: BasicBlocks = field(default_factory=dict)
    __match_args__ = ("name", "params", "blocks")

    def __str__(self):
        result = f"def {self.name}({', '.join(self.params)}):\n"
        for label, statements in self.blocks.items():
            result += '  ' + label + ':\n'
            indent()
            indent()
            result += ''.join(str(s) for s in statements)
            dedent()
            dedent()
        return result


@dataclass
class CFunProgram:
    defs: list[CFunDef]
    __match_args__ = ("defs",)

    def __str__(self):
        return '\n'.join(str(d) for d in self.defs)

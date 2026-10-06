# Slightly modified version of Jeremy Siek's x86_ast.
# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable

from ast_utils import dedent, indent, indent_stmt, label_name


@dataclass
class X86Program:
    """body - a mapping from labels to instr lists, or just a list of instrs of there are no jumps

    entries - the labels that begin a function, each of which gets an .align
    directive.  Only main is .globl; the others are local labels."""
    body: dict[str, list[instr]] | list[instr]
    entries: frozenset[str] = frozenset()

    def __str__(self):
        result = ''
        if isinstance(self.body, dict):
            if 'main' in self.body:
                result += '\t.globl ' + label_name('main') + '\n'
                ss = self.body['main']
                result += '\t.align 16\n'
                result += label_name('main') + ':\n'
                indent()
                result += '\n'.join([str(s) for s in ss]) + '\n'
                result += '\n'
                dedent()
            for (lbl,ss) in [item for item in self.body.items() if item[0] != 'main']:
                if lbl in self.entries:
                    result += '\t.align 16\n'
                result += label_name(lbl) + ':\n'
                indent()
                result += '\n'.join([str(s) for s in ss]) + '\n'
                result += '\n'
                dedent()
        else:
            result += '\t.globl ' + label_name('main') + '\n' + \
                      label_name('main') + ':\n'
            indent()
            result += '\n'.join([str(s) for s in self.body])
            result += '\n'
            dedent()
        result += '\n'
        return result

@dataclass
class X86ProgramDefs:
    defs: list[ast.FunctionDef]

    def __str__(self):
        return "\n".join([str(d) for d in self.defs])

@dataclass
class X86Def:
    """One function's basic blocks, before they are flattened into an
    X86Program.  Register allocation runs once per X86Def."""
    name: str
    blocks: dict[str, list[instr]]
    __match_args__ = ("name", "blocks")

    def __str__(self):
        result = 'def ' + self.name + ':\n'
        for (lbl, ss) in self.blocks.items():
            result += '  ' + lbl + ':\n'
            indent()
            indent()
            result += '\n'.join([str(s) for s in ss]) + '\n'
            dedent()
            dedent()
        return result

class instr: ...
class arg: ...
class location(arg): ...
class defn: ...

@dataclass(frozen=True, eq=False)
class Instr(instr):
    instr: str
    args: tuple[arg, ...]

    def __init__(self, name: str, args: Iterable[arg]):
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, 'instr', name)
        object.__setattr__(self, 'args', tuple(args))

    def source(self):
        return self.args[0]
    def target(self):
        return self.args[-1]
    def __str__(self):
        return indent_stmt() + self.instr + ' ' + ', '.join(str(a) for a in self.args)

@dataclass(frozen=True, eq=False)
class Callq(instr):
    func: str
    num_args: int

    def __str__(self):
        return indent_stmt() + 'callq' + ' ' + label_name(self.func)

@dataclass(frozen=True, eq=False)
class IndirectCallq(instr):
    func: arg
    num_args: int

    def __str__(self):
        return indent_stmt() + 'callq' + ' *' + str(self.func)

@dataclass(frozen=True, eq=False)
class JumpIf(instr):
    cc: str
    label: str

    def __str__(self):
        return indent_stmt() + 'j' + self.cc + ' ' + label_name(self.label)

@dataclass(frozen=True, eq=False)
class Jump(instr):
    label: str

    def __str__(self):
        return indent_stmt() + 'jmp ' + label_name(self.label)

@dataclass(frozen=True, eq=False)
class IndirectJump(instr):
    target: location

    def __str__(self):
        return indent_stmt() + 'jmp *' + str(self.target)

@dataclass(frozen=True, eq=False)
class TailJump(instr):
    func: arg
    arity: int

    def __str__(self):
        return indent_stmt() + 'tailjmp ' + str(self.func)


@dataclass(frozen=True, eq=False)
class Zero(instr):
    size: int

    def __str__(self):
        return indent_stmt() + '.zero ' + str(self.size)

@dataclass(frozen=True)
class Variable(location):
    id: str

    def __str__(self):
        return self.id

@dataclass(frozen=True)
class Immediate(arg):
    value: int

    def __str__(self):
        return '$' +  str(self.value)

@dataclass(frozen=True)
class Reg(location):
    id: str

    def __str__(self):
        return '%' + self.id

@dataclass(frozen=True)
class ByteReg(Reg):
    pass

@dataclass(frozen=True)
class Deref(arg):
    reg: str
    offset: int

    def __str__(self):
        return str(self.offset) + '(%' + self.reg + ')'

@dataclass(frozen=True)
class Global(arg):
    name: str

    def __str__(self):
        return label_name(self.name) + "(%rip)"


def is_memory_arg(a: arg) -> bool:
    match a:
        case Deref(_, _):
            return True
        case Global(_):
            return True
        case _:
            return False

def movq(a: arg, b: arg) -> Instr:
    return Instr('movq', [a, b])

def pushreg(r: str) -> Instr:
    return Instr('pushq', [Reg(r)])

def popreg(r: str) -> Instr:
    return Instr('popq', [Reg(r)])

example1 = X86Program([
    Instr('movq', [Immediate(42), Reg('rax')]),
    Instr('retq', [])
])

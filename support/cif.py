from ast import expr, stmt
from ast_utils import *

BasicBlocks = dict[str, list[stmt]]

@dataclass
class CProgram:
    __match_args__ = ("body",)
    body: dict[str,list[stmt]]

    def __str__(self):
        result = ''
        for (l, ss) in self.body.items():
            result += l + ':\n'
            indent()
            result += ''.join([str(s) for s in ss]) + '\n'
            dedent()
        return result


@dataclass
class CProgramDefs:
    defs: list[stmt]
    __match_args__ = ("defs",)

    def __str__(self):
        return '\n'.join([str(d) for d in self.defs]) + '\n'


@dataclass
class Goto(stmt):
    label: str
    __match_args__ = ("label",)

    def __str__(self):
        return indent_stmt() + 'goto ' + self.label + '\n'

@dataclass
class Begin(expr):
    __match_args__ = ("body", "result")
    body: list[stmt]
    result: expr

    def __str__(self):
        indent()
        stmts = ''.join([str(s) for s in self.body])
        end = indent_stmt() + 'produce ' + str(self.result) + '\n'
        dedent()
        return '{\n' + stmts + end + indent_stmt() + '}'

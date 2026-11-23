"""Interpreter for the source, RCO, and explicate-control forms of Lfun."""
import ast
import operator
import sys
from typing import TextIO

from ast_utils import to_signed
from cfun import CFunDef, CFunProgram, FunDef, FunProgram, split_module
from cif import Begin, Goto


ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
COMPARISONS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}


class LfunInterpreter:
    def __init__(self, program: FunProgram | CFunProgram, stdin: TextIO, stdout: TextIO):
        self.functions: dict[str, ast.FunctionDef | FunDef | CFunDef] = {
            definition.name: definition for definition in program.defs
        }
        self.stdin = stdin
        self.stdout = stdout
        self.output: list[int] = []

    def call(self, name: str, arguments: list[int]) -> int:
        if name == "input_int":
            return to_signed(int(self.stdin.readline()))
        definition = self.functions[name]
        match definition:
            case ast.FunctionDef(args=args, body=body):
                params = [arg.arg for arg in args.args]
            case FunDef(params=params, body=body):
                pass
            case CFunDef(params=params, blocks=blocks):
                body = blocks[name]

        local = dict(zip(params, arguments, strict=True))
        result = self.statements(body, local)
        while isinstance(result, Goto):
            if not isinstance(definition, CFunDef):
                raise ValueError("Unexpected goto outside a basic block")
            result = self.statements(definition.blocks[result.label], local)
        return 0 if result is None else result

    def expression(self, node: ast.expr, local: dict[str, int]) -> int:
        match node:
            case ast.Constant(int() as value):
                return to_signed(value)
            case ast.Name(name):
                return local[name]
            case ast.UnaryOp(ast.USub(), value):
                return to_signed(-self.expression(value, local))
            case ast.UnaryOp(ast.Not(), value):
                return int(not self.expression(value, local))
            case ast.BinOp(left, op, right):
                return to_signed(ARITHMETIC[type(op)](
                    self.expression(left, local), self.expression(right, local)))
            case ast.Compare(left, [op], [right]):
                return int(COMPARISONS[type(op)](
                    self.expression(left, local), self.expression(right, local)))
            case ast.BoolOp(op, values):
                for value in values[:-1]:
                    truth = bool(self.expression(value, local))
                    if isinstance(op, ast.And) and not truth:
                        return 0
                    if isinstance(op, ast.Or) and truth:
                        return 1
                return self.expression(values[-1], local)
            case ast.IfExp(test, body, orelse):
                branch = body if self.expression(test, local) else orelse
                return self.expression(branch, local)
            case ast.Call(ast.Name(name), arguments, []):
                return self.call(name, [self.expression(arg, local) for arg in arguments])
            case Begin(body, result):
                self.statements(body, local)
                return self.expression(result, local)
            case _:
                raise ValueError(f"Unsupported Lfun expression: {node}")

    def statements(self, body: list[ast.stmt], local: dict[str, int]) -> int | Goto | None:
        for statement in body:
            match statement:
                case ast.AnnAssign(_, _, None):
                    pass
                case ast.Assign([ast.Name(name)], value) | ast.AnnAssign(ast.Name(name), _, ast.expr() as value):
                    local[name] = self.expression(value, local)
                case ast.Expr(ast.Call(ast.Name("print"), [arg], [])):
                    value = self.expression(arg, local)
                    self.output.append(value)
                    print(value, file=self.stdout)
                case ast.Expr(value):
                    self.expression(value, local)
                case ast.Return(ast.expr() as value):
                    return self.expression(value, local)
                case Goto():
                    return statement
                case ast.If(test, yes, no):
                    branch = yes if self.expression(test, local) else no
                    result = self.statements(branch, local)
                    if result is not None:
                        return result
                case ast.While(test, loop, []):
                    while self.expression(test, local):
                        result = self.statements(loop, local)
                        if result is not None:
                            return result
                case _:
                    raise ValueError(f"Unsupported Lfun statement: {statement}")
        return None


def interp_lfun(program: ast.Module | FunProgram | CFunProgram, *, stdin=None, stdout=None) -> list[int]:
    if isinstance(program, ast.Module):
        program = split_module(program)
    interpreter = LfunInterpreter(program, stdin or sys.stdin, stdout or sys.stdout)
    interpreter.call("main", [])
    return interpreter.output

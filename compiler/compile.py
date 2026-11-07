# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
import ast
import argparse
import sys

from ast import (Constant, UnaryOp, expr, stmt, Module, Name, IfExp)
from unittest import case
from ast_utils import generate_name  # type: ignore[import-untyped]
from cif import BasicBlocks, Begin, CProgram, Goto  # type: ignore[import-untyped]
from graph import DirectedAdjList, topological_sort, transpose  # type: ignore[import-untyped]
from x86_ast import (X86Program, instr, arg, Variable, Immediate, Instr, Callq, Reg, ByteReg, Jump, JumpIf, Deref, is_memory_arg) # type: ignore[import-untyped]
from x86_interp import checkpoint, add_arguments, interp_x86 # type: ignore[import-untyped]

Binding = tuple[Name, expr]
Temporaries = list[Binding]

SCRATCH_REGISTER = Reg("rax")
MAIN_START = "main_start"
MAIN_CONCLUSION = "main_conclusion"
ARGUMENT_REGISTERS = (
    Reg("rdi"), Reg("rsi"), Reg("rdx"), Reg("rcx"), Reg("r8"), Reg("r9"),
)
CALLEE_SAVED_REGISTERS = (
    Reg("rbx"), Reg("r12"), Reg("r13"), Reg("r14"), Reg("r15"),
)
CALLER_SAVED_REGISTERS = (
    SCRATCH_REGISTER, Reg("rcx"), Reg("rdx"), Reg("rsi"), Reg("rdi"),
    Reg("r8"), Reg("r9"), Reg("r10"), Reg("r11"),
)
COLOR_TO_REGISTER = {
    0: Reg("rcx"), 1: Reg("rdx"), 2: Reg("rsi"), 3: Reg("rdi"),
    4: Reg("r8"), 5: Reg("r9"), 6: Reg("r10"),
    7: Reg("rbx"), 8: Reg("r12"), 9: Reg("r13"), 10: Reg("r14"),
    -1: SCRATCH_REGISTER, -2: Reg("rsp"), -3: Reg("rbp"),
    -4: Reg("r11"), -5: Reg("r15"),
}
REGISTER_TO_COLOR = {register: color for color, register in COLOR_TO_REGISTER.items()}
AVAILABLE_REGISTER_COUNT = sum(color >= 0 for color in COLOR_TO_REGISTER)


class Compiler:

    ############################################################################
    # Shrink
    ############################################################################

    def shrink_exp(self, e: expr) -> expr:
        match e:
            case ast.BoolOp(ast.And(), [left, right, *rest]):
                values = [
                    self.shrink_exp(value)
                    for value in [left, right, *rest]
                ]
                result = values[-1]
                for value in reversed(values[:-1]):
                    result = ast.IfExp(value, result, ast.Constant(False))
                return result
            case ast.BoolOp(ast.Or(), [left, right, *rest]):
                values = [
                    self.shrink_exp(value)
                    for value in [left, right, *rest]
                ]
                result = values[-1]
                for value in reversed(values[:-1]):
                    result = ast.IfExp(value, ast.Constant(True), result)
                return result
            case ast.UnaryOp(ast.Not(), operand):
                return ast.UnaryOp(ast.Not(), self.shrink_exp(operand))
            case ast.UnaryOp(ast.USub(), operand):
                return ast.UnaryOp(ast.USub(), self.shrink_exp(operand))
            case ast.BinOp(
                left,
                (ast.Add() | ast.Sub() | ast.Mult()) as operator,
                right,
            ):
                return ast.BinOp(
                    self.shrink_exp(left),
                    operator,
                    self.shrink_exp(right),
                )
            case ast.Compare(left, [operator], [right]):
                return ast.Compare(
                    self.shrink_exp(left),
                    [operator],
                    [self.shrink_exp(right)],
                )
            case ast.IfExp(test, body, orelse):
                return ast.IfExp(
                    self.shrink_exp(test),
                    self.shrink_exp(body),
                    self.shrink_exp(orelse),
                )
            case ast.Constant() | ast.Name() | ast.Call(ast.Name("input_int"), []):
                return e
            case _:
                raise Exception(f"Unexpected expression during shrink: {e}")

    def shrink_stmt(self, s: stmt) -> list[stmt]:
        match s:
            case ast.Expr(ast.Call(ast.Name("print"), [value])):
                return [ast.Expr(ast.Call(ast.Name("print"), [self.shrink_exp(value)]))]
            case ast.Assign([ast.Name(name)], value):
                return [ast.Assign([ast.Name(name)], self.shrink_exp(value))]
            case ast.If(test, body, orelse):
                shrinked_test = self.shrink_exp(test)
                shrinked_body = [
                    shrunk_stmt
                    for stmt in body
                    for shrunk_stmt in self.shrink_stmt(stmt)
                ]
                shrinked_orelse = [
                    shrunk_stmt
                    for stmt in orelse
                    for shrunk_stmt in self.shrink_stmt(stmt)
                ]
                return [ast.If(shrinked_test, shrinked_body, shrinked_orelse)]
            case _:
                raise Exception(f"Unexpected statement during shrink: {s}")


    def shrink(self, p: Module) -> Module:
        statements: list[stmt] = []
        for statement in p.body:
            statements.extend(self.shrink_stmt(statement))
        return ast.Module(statements)

    ############################################################################
    # Remove Complex Operands
    ############################################################################

    def make_begin(self, temporaries: Temporaries, result: expr) -> expr:
        if not temporaries:
            return result

        return Begin(
            [
                ast.Assign([temporary_name], temporary_value)
                for temporary_name, temporary_value in temporaries
            ],
            result,
        )

    def rco_exp(self, e: expr, need_atomic : bool) -> tuple[expr, Temporaries]:
        match e:
            case ast.Constant() | ast.Name():
                return e, []

            case ast.Call(ast.Name("input_int"), []):
                if not need_atomic:
                    return e, []

                temporary = ast.Name(generate_name("tmp"))
                return temporary, [(temporary, e)]

            case ast.UnaryOp(ast.Not(), operand):
                rco_operand, temporaries = self.rco_exp(
                    operand,
                    need_atomic=not isinstance(operand, ast.Compare),
                )
                result = ast.UnaryOp(ast.Not(), rco_operand)

                if not need_atomic:
                    return result, temporaries

                temporary = ast.Name(generate_name("tmp"))
                return temporary, temporaries + [(temporary, result)]

            case ast.UnaryOp(operator, operand):
                atomic_operand, temporaries = self.rco_exp(operand, True)
                result = ast.UnaryOp(operator, atomic_operand)

                if not need_atomic:
                    return result, temporaries

                temporary = ast.Name(generate_name("tmp"))
                return temporary, temporaries + [(temporary, result)]

            case ast.BinOp(left, operator, right):
                atomic_left, left_temporaries = self.rco_exp(left, True)
                atomic_right, right_temporaries = self.rco_exp(right, True)
                binop = ast.BinOp(atomic_left, operator, atomic_right)

                if not need_atomic:
                    return binop, left_temporaries + right_temporaries

                temporary = ast.Name(generate_name("tmp"))
                return temporary, left_temporaries + right_temporaries + [(temporary, binop)]

            case ast.Compare(left, [operator], [right]):
                atomic_left, left_temporaries = self.rco_exp(left, True)
                atomic_right, right_temporaries = self.rco_exp(right, True)
                comparison = ast.Compare(
                    atomic_left,
                    [operator],
                    [atomic_right],
                )
                temporaries = left_temporaries + right_temporaries

                if not need_atomic:
                    return comparison, temporaries

                temporary = ast.Name(generate_name("tmp"))
                return temporary, temporaries + [(temporary, comparison)]

            case ast.IfExp(test, body, orelse):
                rco_test, test_temporaries = self.rco_exp(test, False)
                rco_body, body_temporaries = self.rco_exp(body, False)
                rco_orelse, orelse_temporaries = self.rco_exp(orelse, False)
                conditional = ast.IfExp(
                    rco_test,
                    self.make_begin(body_temporaries, rco_body),
                    self.make_begin(orelse_temporaries, rco_orelse),
                )

                if not need_atomic:
                    return conditional, test_temporaries

                temporary = ast.Name(generate_name("tmp"))
                return temporary, test_temporaries + [(temporary, conditional)]
            case _:
                raise Exception(f"Unexpected expression during RCO: {e}")

    def rco_stmt(self, s: stmt) -> list[stmt]:
        match s:
            case ast.Expr(ast.Call(ast.Name("print"), [value])):
                atom, temporaries = self.rco_exp(value, True)
                statements: list[stmt] = [
                    ast.Assign([name], temporary)
                    for name, temporary in temporaries
                ]
                statements.append(ast.Expr(ast.Call(ast.Name("print"), [atom])))
                return statements
            case ast.Assign([ast.Name(name)], value):
                expression, temporaries = self.rco_exp(value, False)
                statements = [
                    ast.Assign([temporary_name], temporary_value)
                    for temporary_name, temporary_value in temporaries
                ]
                statements.append(ast.Assign([ast.Name(name)], expression))
                return statements
            case ast.If(test, body, orelse):
                expression, temporaries = self.rco_exp(test, False)
                rco_test = self.make_begin(temporaries, expression)
                rco_body = [
                    rco_statement
                    for statement in body
                    for rco_statement in self.rco_stmt(statement)
                ]
                rco_orelse = [
                    rco_statement
                    for statement in orelse
                    for rco_statement in self.rco_stmt(statement)
                ]
                return [ast.If(rco_test, rco_body, rco_orelse)]
            case _:
                raise Exception(f"Unexpected statement during RCO: {s}")

    def remove_complex_operands(self, p: Module) -> Module:
        statements: list[stmt] = []
        for statement in p.body:
            statements.extend(self.rco_stmt(statement))
        return ast.Module(statements)

    ############################################################################
    # ANF Check
    ############################################################################

    def is_atom(self, e: expr) -> bool:
        return isinstance(e, (ast.Name, ast.Constant))

    def is_anf_exp(self, e: expr) -> bool:
        match e:
            case ast.Constant() | ast.Name():
                return True
            case ast.BinOp(left, _, right):
                return self.is_atom(left) and self.is_atom(right)
            case ast.UnaryOp(ast.Not(), ast.Compare() as comparison):
                return self.is_anf_exp(comparison)
            case ast.UnaryOp(_, operand):
                return self.is_atom(operand)
            case ast.Compare(left, [_], [right]):
                return self.is_atom(left) and self.is_atom(right)
            case ast.Call(_, arguments):
                return all(self.is_atom(argument) for argument in arguments)
            case ast.IfExp(test, body, orelse):
                return (
                    self.is_anf_exp(test)
                    and self.is_anf_exp(body)
                    and self.is_anf_exp(orelse)
                )
            case Begin() as begin:
                return all(
                    self.is_anf_stmt(statement) for statement in begin.body
                ) and self.is_anf_exp(begin.result)
            case _:
                return False

    def is_anf_stmt(self, s: stmt) -> bool:
        match s:
            case ast.Expr(value):
                return self.is_anf_exp(value)
            case ast.Assign(_, value):
                return self.is_anf_exp(value)
            case ast.If(test, body, orelse):
                return (
                    self.is_anf_exp(test)
                    and all(self.is_anf_stmt(statement) for statement in body)
                    and all(self.is_anf_stmt(statement) for statement in orelse)
                )
            case _:
                return False

    def is_anf(self, p: Module) -> bool:
        for statement in p.body:
            if not self.is_anf_stmt(statement):
                return False
        return True

    ############################################################################
    # Explicate Control
    ############################################################################

    def create_block(
        self,
        statements: list[stmt],
        basic_blocks: BasicBlocks,
    ) -> list[stmt]:
        match statements:
            case [Goto(_label)]:
                return statements
            case _:
                label = generate_name("block")
                basic_blocks[label] = statements
                return [Goto(label)]

    def explicate_stmts(
        self,
        statements: list[stmt],
        cont: list[stmt],
        basic_blocks: BasicBlocks,
    ) -> list[stmt]:
        for statement in reversed(statements):
            cont = self.explicate_stmt(statement, cont, basic_blocks)
        return cont

    def explicate_stmt(
        self,
        s: stmt,
        cont: list[stmt],
        basic_blocks: BasicBlocks,
    ) -> list[stmt]:
        match s:
            case ast.Expr(ast.Call(ast.Name("print"), [atom]) as call):
                return self.explicate_effect(call, cont, basic_blocks)
            case ast.Assign([lhs], rhs):
                return self.explicate_assign(rhs, lhs, cont, basic_blocks)
            case ast.If(test, body, orelse):
                goto_cont = self.create_block(cont, basic_blocks)
                new_body   = self.explicate_stmts(body,   goto_cont, basic_blocks)
                new_orelse = self.explicate_stmts(orelse, goto_cont, basic_blocks)
                return self.explicate_pred(test, new_body, new_orelse, basic_blocks)
            case _:
                raise Exception(f"Unexpected statement during explicate control: {s}")

    def explicate_assign(
        self,
        rhs: expr,
        lhs: expr,
        cont: list[stmt],
        basic_blocks: BasicBlocks,
    ) -> list[stmt]:
        match rhs:
            case IfExp(test, body, orelse):
                goto_cont = self.create_block(cont, basic_blocks)
                then_branch = self.explicate_assign(body,   lhs, goto_cont, basic_blocks)
                else_branch = self.explicate_assign(orelse, lhs, goto_cont, basic_blocks)
                return self.explicate_pred(test, then_branch, else_branch, basic_blocks)
            case Begin() as begin:
                result_continuation = self.explicate_assign(begin.result, lhs, cont, basic_blocks)
                return self.explicate_stmts(begin.body, result_continuation, basic_blocks)
            case (
                ast.Name()
                | ast.Constant(int() | bool())
                | ast.Call(ast.Name("input_int"), [], [])
                | ast.UnaryOp(ast.USub() | ast.Not(), _)
                | ast.BinOp(_, ast.Add() | ast.Sub() | ast.Mult(), _)
                | ast.Compare(_, [_], [_])
            ) if self.is_anf_exp(rhs):
                return [ast.Assign([lhs], rhs)] + cont
            case _:
                raise Exception(f"Unexpected assignment expression during explicate control: {rhs}")

    def explicate_effect(
        self,
        e: expr,
        cont: list[stmt],
        basic_blocks: BasicBlocks,
    ) -> list[stmt]:
        match e:
            case Begin() as begin:
                result_continuation = self.explicate_effect(begin.result, cont, basic_blocks)
                return self.explicate_stmts(begin.body, result_continuation, basic_blocks)
            case ast.Call():
                return [ast.Expr(e)] + cont
            case _:
                raise Exception(f"Unexpected effect expression during explicate control: {e}")

    def explicate_pred(
        self,
        cnd: expr,
        thn: list[stmt],
        els: list[stmt],
        basic_blocks: BasicBlocks,
    ) -> list[stmt]:
        match cnd:
            case ast.Compare(left, [op], [right]):
                goto_thn = self.create_block(thn, basic_blocks)
                goto_els = self.create_block(els, basic_blocks)
                return [ast.If(cnd, goto_thn, goto_els)]

            case ast.Constant(True):  return thn
            case ast.Constant(False): return els

            case ast.UnaryOp(ast.Not(), operand):
                return self.explicate_pred(operand, els, thn, basic_blocks)

            case ast.IfExp(test, body, orelse):
                elsblock = self.create_block(els, basic_blocks)
                thnblock = self.create_block(thn, basic_blocks)
                body_branch   = self.explicate_pred(body,   thnblock, elsblock, basic_blocks)
                orelse_branch = self.explicate_pred(orelse, thnblock, elsblock, basic_blocks)
                return self.explicate_pred(test, body_branch, orelse_branch, basic_blocks)

            case Begin() as begin:
                new_result = self.explicate_pred(
                    begin.result, thn, els, basic_blocks
                )
                return self.explicate_stmts(
                    begin.body, new_result, basic_blocks
                )

            case ast.Name() | ast.Constant():
                return [ast.If(ast.Compare(cnd, [ast.Eq()], [ast.Constant(False)]),
                        self.create_block(els, basic_blocks),
                        self.create_block(thn, basic_blocks))]
            case _:
                temp = ast.Name(generate_name('tmp'))
                return [ast.Assign([temp], cnd)] + self.explicate_pred(temp, thn, els,
                                                            basic_blocks)

    def explicate_control(self, p: Module) -> CProgram:
        basic_blocks: BasicBlocks = {}
        basic_blocks[MAIN_START] = self.explicate_stmts(
            p.body,
            [ast.Return(ast.Constant(0))],
            basic_blocks,
        )
        return CProgram(basic_blocks)

    ############################################################################
    # Select Instructions
    ############################################################################

    def immediate_size(self, value: int) -> int:
        if -(2 ** 31) <= value < 2 ** 31:
            return 32
        if -(2 ** 63) <= value < 2 ** 63:
            return 64
        raise Exception(f"Immediate value {value} does not fit in 64 bits")

    # The expression e passed to select_arg should furthermore be an atom.
    # (But there is no type for atoms, so the type of e is given as expr.)
    def select_arg(self, e: expr) -> arg:
        match e:
            case Constant(value):
                match value:
                    case bool():
                        return Immediate(1 if value else 0)
                    case int():
                        self.immediate_size(value)
                        return Immediate(value)
                    case _:
                        raise Exception(f"Only integers allowed {e}")
            case Name(id):
                return Variable(id)
            case _:
                raise Exception(f"Unexpected expression {e}")

    def select_jmp(self, comparison: ast.cmpop) -> str:
        match comparison:
            case ast.Eq():
                return "e"
            case ast.NotEq():
                return "ne"
            case ast.Lt():
                return "l"
            case ast.LtE():
                return "le"
            case ast.Gt():
                return "g"
            case ast.GtE():
                return "ge"
            case _:
                raise Exception(f"Unsupported comparison: {comparison}")

    def negate_comparison(self, comparison: ast.cmpop) -> ast.cmpop:
        match comparison:
            case ast.Eq():
                return ast.NotEq()
            case ast.NotEq():
                return ast.Eq()
            case ast.Lt():
                return ast.GtE()
            case ast.LtE():
                return ast.Gt()
            case ast.Gt():
                return ast.LtE()
            case ast.GtE():
                return ast.Lt()
            case _:
                raise Exception(f"Unsupported comparison: {comparison}")

    def compare_constants(
        self, comparison: ast.cmpop, left: int, right: int
    ) -> bool:
        self.immediate_size(left)
        self.immediate_size(right)
        match comparison:
            case ast.Eq():
                return left == right
            case ast.NotEq():
                return left != right
            case ast.Lt():
                return left < right
            case ast.LtE():
                return left <= right
            case ast.Gt():
                return left > right
            case ast.GtE():
                return left >= right
            case _:
                raise Exception(f"Unsupported comparison: {comparison}")

    def select_compare(
        self, var: expr, comparison: ast.cmpop, left: expr, right: expr
    ) -> list[instr]:
        left_arg = self.select_arg(left)
        right_arg = self.select_arg(right)
        match left_arg, right_arg:
            case Immediate(left_value), Immediate(right_value):
                value = self.compare_constants(comparison, left_value, right_value)
                return [Instr("movq", [Immediate(int(value)), self.select_arg(var)])]
        al = ByteReg("al")
        return [
            Instr("cmpq", [right_arg, left_arg]),
            Instr("set" + self.select_jmp(comparison), [al]),
            Instr("movzbq", [al, self.select_arg(var)]),
        ]

    def select_stmt(self, s: stmt) -> list[instr]:
        match s:
            case Goto(label):
                return [Jump(label)]

            case ast.Return(value):
                if value is None:
                    return [Jump(MAIN_CONCLUSION)]
                return [
                    Instr("movq", [self.select_arg(value), SCRATCH_REGISTER]),
                    Jump(MAIN_CONCLUSION),
                ]

            case ast.If(ast.Compare(left, [comparison], [right]), [Goto(thn)], [Goto(els)]):
                left_arg = self.select_arg(left)
                right_arg = self.select_arg(right)
                match left_arg, right_arg:
                    case Immediate(left_value), Immediate(right_value):
                        constant_result = self.compare_constants(
                            comparison, left_value, right_value
                        )
                        return [Jump(thn if constant_result else els)]
                return [
                    Instr("cmpq", [right_arg, left_arg]),
                    JumpIf(self.select_jmp(comparison), thn),
                    Jump(els),
                ]

            case ast.Expr(ast.Call(ast.Name("print"), [atom])):
                return [
                    Instr("movq", [self.select_arg(atom), Reg("rdi")]),
                    Callq("print_int", 1),
                ]

            case ast.Assign(targets=[ast.Name(lhs)], value=rhs):
                match rhs:
                    case ast.Constant() | ast.Name():
                        return [Instr("movq", [self.select_arg(rhs), Variable(lhs)])]

                    case ast.Call(ast.Name("input_int"), []):
                        return [
                            Callq("input_int", 0),
                            Instr("movq", [Reg("rax"), Variable(lhs)]),
                        ]

                    case ast.UnaryOp(ast.Not(), ast.Compare(left, [comparison], [right])):
                        return self.select_compare(
                            ast.Name(lhs),
                            self.negate_comparison(comparison),
                            left,
                            right,
                        )

                    case ast.Compare(left, [comparison], [right]):
                        return self.select_compare(
                            ast.Name(lhs), comparison, left, right
                        )

                    case ast.UnaryOp(ast.Not(), atom):
                        source = self.select_arg(atom)
                        target = Variable(lhs)
                        flip = Instr("xorq", [Immediate(1), target])
                        if source == target:
                            return [flip]
                        return [Instr("movq", [source, target]), flip]

                    case ast.UnaryOp(ast.USub(), value):
                        return [
                            Instr("movq", [self.select_arg(value), Variable(lhs)]),
                            Instr("negq", [Variable(lhs)]),
                        ]
                    case ast.BinOp(left, ast.Add(), right):
                        if type(left) is ast.Name and left.id == lhs:
                            return [
                                Instr("addq", [self.select_arg(right), Variable(lhs)]),
                            ]
                        elif type(right) is ast.Name and right.id == lhs:
                            return [
                                Instr("addq", [self.select_arg(left), Variable(lhs)]),
                            ]
                        else:
                            return [
                                Instr("movq", [self.select_arg(left), Variable(lhs)]),
                                Instr("addq", [self.select_arg(right), Variable(lhs)]),
                            ]
                    case ast.BinOp(left, ast.Sub(), right):
                        if type(left) is ast.Name and left.id == lhs:
                            return [
                                Instr("subq", [self.select_arg(right), Variable(lhs)]),
                            ]
                        elif type(right) is ast.Name and right.id == lhs:
                            return [
                                Instr("negq", [Variable(lhs)]),
                                Instr("addq", [self.select_arg(left), Variable(lhs)]),
                            ]
                        else:
                            return [
                                Instr("movq", [self.select_arg(left), Variable(lhs)]),
                                Instr("subq", [self.select_arg(right), Variable(lhs)]),
                            ]
                    case ast.BinOp(left, ast.Mult(), right):
                        left_arg = self.select_arg(left)
                        right_arg = self.select_arg(right)
                        if isinstance(left_arg, Immediate) and isinstance(right_arg, Immediate):
                            product = left_arg.value * right_arg.value
                            self.immediate_size(product)
                            return [
                                Instr("movq", [Immediate(product), Variable(lhs)])
                            ]

                        if isinstance(right_arg, Immediate):
                            left_arg, right_arg = right_arg, left_arg

                        # Right arg is not immadiete, emitted imulq is correct.
                        return [
                            Instr("movq", [left_arg, SCRATCH_REGISTER]),
                            Instr("imulq", [right_arg, SCRATCH_REGISTER]),
                            Instr("movq", [SCRATCH_REGISTER, Variable(lhs)]),
                        ]
                    case _:
                        raise Exception(f"Unrecognized right-hand side of assignment: {rhs}")
            case _:
                raise Exception(f"Unexpected statement {s}, it must be assignment or print")

    def select_instructions(self, p: CProgram) -> X86Program:
        blocks: dict[str, list[instr]] = {}
        for label, statements in p.body.items():
            instructions: list[instr] = []
            for statement in statements:
                instructions.extend(self.select_stmt(statement))
            blocks[label] = instructions
        return X86Program(blocks)


    ############################################################################
    # Register Allocation
    ############################################################################

    def operand_locations(self, operand: arg) -> set[arg]:
        match operand:
            case ByteReg("al"):
                return {Reg("rax")}
            case Variable() | Reg():
                return {operand}
            case Immediate():
                return set()
            case _:
                raise ValueError(f"Unexpected register allocation operand: {operand}")

    def instruction_read(self, instruction: instr) -> set[arg]:
        match instruction:
            case Callq(num_args=arity):
                return set(ARGUMENT_REGISTERS[:arity])
            case Instr(instr="movq" | "movzbq", args=[source, _]):
                return self.operand_locations(source)
            case Instr(
                instr="addq" | "subq" | "imulq" | "cmpq" | "xorq",
                args=[source, destination],
            ):
                return self.operand_locations(source) | self.operand_locations(destination)
            case Instr(instr="negq", args=[destination]):
                return self.operand_locations(destination)
            case Instr(instr=name) if name.startswith("set"):
                return set()
            case Jump() | JumpIf():
                return set()
            case _:
                raise ValueError(f"Reads are not defined for {instruction}")

    def instruction_write(self, instruction: instr) -> set[arg]:
        match instruction:
            case Callq():
                return set(CALLER_SAVED_REGISTERS)
            case Instr(
                instr="movq" | "movzbq" | "addq" | "subq" | "imulq" | "xorq",
                args=[_, destination],
            ) | Instr(instr="negq", args=[destination]):
                return self.operand_locations(destination)
            case Instr(instr=name, args=[destination]) if name.startswith("set"):
                return self.operand_locations(destination)
            case Instr(instr="cmpq") | Jump() | JumpIf():
                return set()
            case _:
                raise ValueError(f"Writes are not defined for {instruction}")

    def control_flow_graph(self, p: X86Program) -> DirectedAdjList:
        assert isinstance(p.body, dict)
        graph = DirectedAdjList()
        for label, instructions in p.body.items():
            graph.add_vertex(label)
            for instruction in instructions:
                match instruction:
                    case Jump(label=target) | JumpIf(label=target):
                        graph.add_edge(label, target)
        return graph

    def uncover_live(self, p: X86Program) -> dict[str, list[set[arg]]]:
        blocks = p.body
        assert isinstance(blocks, dict)
        order = topological_sort(transpose(self.control_flow_graph(p)))
        live_before_block: dict[str, set[arg]] = {}
        live_after: dict[str, list[set[arg]]] = {}
        for label in order:
            instructions = blocks.get(label, [])
            live: set[arg] = set()
            after_instructions: list[set[arg]] = [set() for _ in instructions]
            for index in reversed(range(len(instructions))):
                instruction = instructions[index]
                match instruction:
                    case Jump(label=target):
                        after = live_before_block[target].copy()
                    case JumpIf(label=target):
                        after = live_before_block[target] | live
                    case _:
                        after = live.copy()
                after_instructions[index] = after
                live = (
                    (after - self.instruction_write(instruction))
                    | self.instruction_read(instruction)
                )
            live_after[label] = after_instructions
            live_before_block[label] = live
        return live_after

    def instruction_edges(
        self,
        instruction: instr,
        live_after: set[arg],
    ) -> set[tuple[arg, arg]]:
        writes = self.instruction_write(instruction)
        match instruction:
            case Instr(instr="movq", args=[source, _]):
                neighbors = live_after - {source}
            case _:
                neighbors = live_after
        return {
            (destination, neighbor)
            for destination in writes
            for neighbor in neighbors
            if neighbor != destination
        }

    def build_interference_graph(
        self,
        p: X86Program,
        live_after: dict[str, list[set[arg]]],
    ) -> dict[arg, set[arg]]:
        blocks = p.body
        assert isinstance(blocks, dict)
        graph: dict[arg, set[arg]] = {}
        for label, instructions in blocks.items():
            for index, instruction in enumerate(instructions):
                writes = self.instruction_write(instruction)
                for location in live_after[label][index] | writes:
                    graph.setdefault(location, set())

                edges = self.instruction_edges(instruction, live_after[label][index])
                for destination, neighbor in edges:
                    graph[destination].add(neighbor)
                    graph[neighbor].add(destination)
        return graph


    def debug_register_allocation(
        self,
        p: X86Program,
        live_after: dict[str, list[set[arg]]],
        allocatable_registers: tuple[Reg, ...],
        coloring: dict[arg, int],
        homes: dict[Variable, arg],
    ) -> None:
        blocks = p.body
        assert isinstance(blocks, dict)

        print(
            "Available for allocation: "
            + ", ".join(map(str, allocatable_registers)),
            file=sys.stderr,
        )

        rows: list[tuple[str, str, str]] = []
        for label, instructions in blocks.items():
            for index, instruction in enumerate(instructions):
                after = live_after[label][index]
                live_text = "{" + ", ".join(sorted(map(str, after))) + "}"
                edges = self.instruction_edges(instruction, after)
                edge_text = ", ".join(sorted(
                    f"{destination}-{neighbor}" for destination, neighbor in edges
                )) or "--"
                rows.append((f"{label}: {str(instruction).strip()}", live_text, edge_text))

        instruction_width = max([len("Instruction")] + [len(row[0]) for row in rows])
        live_width = max([len("live-after")] + [len(row[1]) for row in rows])
        print(file=sys.stderr)
        print(
            f"  {'Instruction':<{instruction_width}}  "
            f"{'live-after':<{live_width}}  edges added",
            file=sys.stderr,
        )
        for instruction_text, live_text, edge_text in rows:
            print(
                f"  {instruction_text:<{instruction_width}}  "
                f"{live_text:<{live_width}}  {edge_text}",
                file=sys.stderr,
            )

        print(file=sys.stderr)
        print("Colors and homes:", file=sys.stderr)
        for location in sorted(coloring, key=str):
            home = f" -> {homes[location]}" if isinstance(location, Variable) else ""
            print(f"  {location}: {coloring[location]}{home}", file=sys.stderr)
        print(file=sys.stderr)

    def color_graph(
        self, graph: dict[arg, set[arg]], coloring: dict[arg, int]
    ) -> None:
        uncolored = set(graph) - coloring.keys()
        while uncolored:
            vertex = max(
                uncolored,
                key=lambda v: len({coloring[n] for n in graph[v] if n in coloring}),
            )
            used = {coloring[n] for n in graph[vertex] if n in coloring}
            color = 0
            while color in used:
                color += 1
            coloring[vertex] = color
            uncolored.remove(vertex)

    def allocate_registers(
        self,
        p: X86Program,
        registers_cap: int = AVAILABLE_REGISTER_COUNT,
        debug_allocate_registers: bool = False,
    ) -> X86Program:
        if registers_cap < 1:
            raise ValueError("registers_cap must be at least 1")
        blocks = p.body
        assert isinstance(blocks, dict)
        live_after = self.uncover_live(p)
        graph = self.build_interference_graph(p, live_after)

        registers_cap = min(registers_cap, AVAILABLE_REGISTER_COUNT)
        coloring: dict[arg, int] = {
            register: REGISTER_TO_COLOR[register]
            for register in graph if isinstance(register, Reg)
        }
        self.color_graph(graph, coloring)

        homes: dict[Variable, arg] = {}
        stack_offsets: dict[int, int] = {}
        for location, color in coloring.items():
            if not isinstance(location, Variable):
                continue
            if color < registers_cap:
                homes[location] = COLOR_TO_REGISTER[color]
            else:
                if color not in stack_offsets:
                    stack_offsets[color] = -8 * (len(stack_offsets) + 1)
                homes[location] = Deref("rbp", stack_offsets[color])

        if debug_allocate_registers:
            self.debug_register_allocation(
                p, live_after,
                tuple(COLOR_TO_REGISTER[color] for color in range(registers_cap)),
                coloring, homes,
            )
        p.body = {
            label: [
                Instr(
                    instruction.instr,
                    [
                        homes[operand] if isinstance(operand, Variable) else operand
                        for operand in instruction.args
                    ],
                )
                if isinstance(instruction, Instr) else instruction
                for instruction in instructions
            ]
            for label, instructions in blocks.items()
        }
        return p

    ############################################################################
    # Patch Instructions
    ############################################################################
    def patch_instr(self, i: instr) -> list[instr]:
        # Keep the scratch register out of the allocatable register set.
        match i:
            case Instr(instr="movq", args=[source, destination]) if source == destination:
                return []
            case Instr(instr="cmpq", args=[Immediate(), Immediate()]):
                raise AssertionError("constant comparison should be selected without cmpq")
            case Instr(instr="cmpq", args=[source, Immediate() as destination]):
                self.immediate_size(destination.value)
                return [
                    Instr("movq", [destination, SCRATCH_REGISTER]),
                    Instr("cmpq", [source, SCRATCH_REGISTER]),
                ]
            case Instr(instr="movzbq", args=[source, destination]) \
                    if is_memory_arg(destination):
                return [
                    Instr("movzbq", [source, SCRATCH_REGISTER]),
                    Instr("movq", [SCRATCH_REGISTER, destination]),
                ]
            case Instr():
                for a in i.args:
                    if type(a) is Immediate:
                        self.immediate_size(a.value)

                match i.args:
                    case [Immediate(value), destination]:
                        size = self.immediate_size(value)
                        if size == 32:
                            return [i]
                        if i.instr == "movq" and type(destination) is Reg:
                            return [i]
                        return [
                            Instr("movq", [Immediate(value), SCRATCH_REGISTER]),
                            Instr(i.instr, [SCRATCH_REGISTER, destination]),
                        ]

                    case [source, destination] \
                            if is_memory_arg(source) and is_memory_arg(destination):
                        return [
                            Instr("movq", [source, SCRATCH_REGISTER]),
                            Instr(i.instr, [SCRATCH_REGISTER, destination]),
                        ]
        return [i]

    def patch_instructions(self, p: X86Program) -> X86Program:
        assert isinstance(p.body, dict)
        blocks: dict[str, list[instr]] = {}
        for label, instructions in p.body.items():
            blocks[label] = [
                patched
                for instruction in instructions
                for patched in self.patch_instr(instruction)
            ]
        return X86Program(blocks, p.entries)

    ############################################################################
    # Prelude & Conclusion
    ############################################################################

    def used_callee_saved_registers(self, instructions: list[instr]) -> set[Reg]:
        return {
            operand
            for instruction in instructions if isinstance(instruction, Instr)
            for operand in instruction.args
            if isinstance(operand, Reg) and operand in CALLEE_SAVED_REGISTERS
        }

    def used_stack_size(self, instructions: list[instr]) -> int:
        return max(
            (
                -operand.offset
                for instruction in instructions if isinstance(instruction, Instr)
                for operand in instruction.args
                if isinstance(operand, Deref) and operand.reg == "rbp"
            ),
            default=0,
        )

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        blocks = p.body
        assert isinstance(blocks, dict)
        instructions = [i for block in blocks.values() for i in block]
        used_stack_bytes = self.used_stack_size(instructions)

        used_saved_registers = self.used_callee_saved_registers(instructions)
        saved_registers = [
            register for register in CALLEE_SAVED_REGISTERS
            if register in used_saved_registers
        ]
        saved_register_bytes = 8 * len(saved_registers)
        stack_adjustment = (
            (used_stack_bytes + saved_register_bytes + 15) // 16 * 16
            - saved_register_bytes
        )

        prologue: list[instr] = [
            Instr("pushq", [Reg("rbp")]),
            Instr("movq", [Reg("rsp"), Reg("rbp")]),
        ]
        if stack_adjustment:
            prologue.append(Instr("subq", [Immediate(stack_adjustment), Reg("rsp")]))
        prologue.extend(Instr("pushq", [register]) for register in saved_registers)

        conclusion: list[instr] = [
            Instr("popq", [register]) for register in reversed(saved_registers)
        ]
        if stack_adjustment:
            conclusion.append(Instr("addq", [Immediate(stack_adjustment), Reg("rsp")]))
        conclusion.extend([
            Instr("popq", [Reg("rbp")]),
            Instr("retq", []),
        ])

        p.body = {
            "main": prologue + [Jump(MAIN_START)],
            **blocks,
            MAIN_CONCLUSION: conclusion,
        }

        return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--debug-allocate-registers", action="store_true", help="print live-after sets, added edges, colors, and homes")
    parser.add_argument(
        "--registers-cap", type=int, default=AVAILABLE_REGISTER_COUNT,
        help=f"number of allocatable registers (at least 1; {AVAILABLE_REGISTER_COUNT} available)",
    )
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.input, "r") as file:
        module = ast.parse(file.read())

    compiler = Compiler()

    shrinked = compiler.shrink(module)
    checkpoint(args, "shrink", shrinked)
    simplified = compiler.remove_complex_operands(shrinked)
    assert compiler.is_anf(simplified), \
        f"The AST is not in A-normal form:\n{simplified}"
    checkpoint(args, "remove_complex_operands", simplified)
    explicated = compiler.explicate_control(simplified)
    checkpoint(args, "explicate_control", explicated)
    instruction_list = compiler.select_instructions(explicated)
    checkpoint(args, "select_instructions", instruction_list)
    allocated = compiler.allocate_registers(
        instruction_list, registers_cap=args.registers_cap,
        debug_allocate_registers=args.debug_allocate_registers,
    )
    checkpoint(args, "allocate_registers", allocated)
    patched_instructions = compiler.patch_instructions(allocated)
    checkpoint(args, "patch_instructions", patched_instructions)
    added_prelude_conclusion = compiler.prelude_and_conclusion(patched_instructions)
    checkpoint(args, "prelude_and_conclusion", added_prelude_conclusion)

    if args.interp:
        interp_x86(added_prelude_conclusion)
        return

    if args.output is None:
        print(added_prelude_conclusion)
    else:
        with open(args.output, "w") as output_file:
            output_file.write(str(added_prelude_conclusion))

if __name__ == "__main__":
    main()

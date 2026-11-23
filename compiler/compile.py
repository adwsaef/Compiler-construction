# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
import ast
import argparse
import sys
from collections import deque
from copy import replace
from dataclasses import dataclass

from ast import (Constant, expr, stmt, Module, Name, IfExp)
from ast_utils import generate_name  # type: ignore[import-untyped]
from cfun import CFunDef, CFunProgram, FunDef, FunProgram  # type: ignore[import-untyped]
from cif import BasicBlocks, Begin, Goto  # type: ignore[import-untyped]
from graph import DirectedAdjList, transpose  # type: ignore[import-untyped]
from x86_ast import (X86Def, X86Program, instr, arg, Variable, Immediate, Instr, Callq, Reg, ByteReg, Jump, JumpIf, Deref, is_memory_arg) # type: ignore[import-untyped]
from x86_interp import NOT_MODELLED, RUNTIME, checkpoint, add_arguments, interp_x86, to_signed # type: ignore[import-untyped]

Binding = tuple[Name, expr]
Temporaries = list[Binding]

SCRATCH_REGISTER = Reg("rax")
COMPARISON_OPERATORS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
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


ScalarTypes = frozenset[type[int]]
TypeEnvironment = dict[str, ScalarTypes]
INT_TYPE: ScalarTypes = frozenset({int})
BOOL_TYPE: ScalarTypes = frozenset({bool})
SCALAR_TYPES = INT_TYPE | BOOL_TYPE
NO_TYPES: ScalarTypes = frozenset()


@dataclass
class Signature:
    parameters: dict[str, ScalarTypes]
    result: ScalarTypes


@dataclass
class TypeScope:
    functions: dict[str, Signature]
    variables: TypeEnvironment
    function_name: str


BUILTINS = {
    "input_int": Signature({}, INT_TYPE),
    "print": Signature({"value": SCALAR_TYPES}, NO_TYPES),
}


def function_label(name: str) -> str:
    if name in RUNTIME or name in NOT_MODELLED:
        return f"fun.{name}"
    return name


class Compiler:
    ############################################################################
    # Split Module
    ############################################################################

    def split_module(self, p: Module) -> FunProgram[ast.FunctionDef]:
        definitions: dict[str, ast.FunctionDef] = {}
        main_body: list[stmt] = []
        for statement in p.body:
            match statement:
                case ast.FunctionDef(name=name):
                    if name == "main" or name in BUILTINS:
                        raise ValueError(f"Reserved function name: {name}")
                    if name in definitions:
                        raise ValueError(f"Duplicate function definition: {name}")
                    definitions[name] = statement
                case ast.Import() | ast.ImportFrom():
                    pass
                case _:
                    main_body.append(statement)
        main_body.append(ast.Return(ast.Constant(0)))
        definitions["main"] = ast.FunctionDef(
            name="main", args=ast.arguments(), body=main_body, returns=ast.Name(id="int"),
        )
        return FunProgram(list(definitions.values()))

    ############################################################################
    # Check Functions
    ############################################################################

    def function_arity(self, definition: ast.FunctionDef) -> int:
        name = definition.name
        if definition.decorator_list or definition.type_params:
            raise ValueError(f"Decorators and type parameters are not supported: {name}")

        args = definition.args
        if (
            args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or args.defaults or args.kw_defaults
        ):
            raise NotImplementedError(f"{name} only supports ordinary parameters without defaults")
        params = [parameter.arg for parameter in args.args]

        if len(set(params)) != len(params):
            raise ValueError(f"Duplicate parameter in function: {name}")

        max_parameters = len(ARGUMENT_REGISTERS)
        if len(params) > max_parameters:
            raise ValueError(f"{name} supports at most {max_parameters} parameters")

        return len(params)

    def check_function_uses(self, node: ast.AST, arities: dict[str, int]) -> None:
        match node:
            case ast.Name(name) | ast.arg(name):
                if name in arities:
                    raise TypeError(f"Function name '{name}' cannot be used as a variable")
            case ast.Call(ast.Name(name), arguments, []):
                if name not in arities:
                    raise TypeError(f"Unknown function '{name}'")
                arity = arities[name]
                if len(arguments) != arity:
                    raise TypeError(f"Function '{name}' expects {arity} arguments, got {len(arguments)}")
                for argument in arguments:
                    self.check_function_uses(argument, arities)
            case ast.Call():
                raise TypeError("Only direct calls with positional arguments are supported")
            case ast.AnnAssign(target, _, value):
                self.check_function_uses(target, arities)
                if value is not None:
                    self.check_function_uses(value, arities)
            case _:
                for child in ast.iter_child_nodes(node):
                    self.check_function_uses(child, arities)

    def check_functions(self, p: FunProgram[ast.FunctionDef]) -> None:
        arities = {name: len(sig.parameters) for name, sig in BUILTINS.items()}
        for definition in p.defs:
            arities[definition.name] = self.function_arity(definition)

        for definition in p.defs:
            for parameter in definition.args.args:
                self.check_function_uses(parameter, arities)
            for statement in definition.body:
                self.check_function_uses(statement, arities)

    ############################################################################
    # Type Check
    ############################################################################

    def annotation_type(self, annotation: expr | None, location: str) -> ScalarTypes:
        match annotation:
            case ast.Name("int"):
                return INT_TYPE
            case ast.Name("bool"):
                return BOOL_TYPE
            case None:
                raise TypeError(f"Missing type annotation for {location}")
            case _:
                raise TypeError(f"Unsupported type annotation '{annotation}'")

    def variable_type(self, name: str, scope: TypeScope) -> ScalarTypes:
        declared = scope.variables.get(name)
        if declared is None:
            raise TypeError(f"Missing type annotation for variable '{name}'")
        return declared

    def require_type(
        self, actual: ScalarTypes, expected: ScalarTypes, location: str,
    ) -> None:
        if actual == NO_TYPES:
            raise TypeError(f"{location} requires a value")
        if actual - expected:
            expected_name = " or ".join(sorted(t.__name__ for t in expected))
            actual_names = " or ".join(sorted(t.__name__ for t in actual))
            raise TypeError(f"{location} expects {expected_name}, got {actual_names}")

    def type_check_call(self, name: str, arguments: list[expr], scope: TypeScope) -> ScalarTypes:
        signature = scope.functions[name]
        for index, (argument, expected) in enumerate(zip(arguments, signature.parameters.values()), 1):
            received = self.type_check_exp(argument, scope)
            self.require_type(received, expected, f"Argument {index} of '{name}'")
        return signature.result

    def type_check_exp(self, e: expr, scope: TypeScope) -> ScalarTypes:
        match e:
            case ast.Constant(int() as value):
                return BOOL_TYPE if isinstance(value, bool) else INT_TYPE
            case ast.Name(name):
                return self.variable_type(name, scope)
            case ast.UnaryOp(ast.Not(), operand):
                self.require_type(self.type_check_exp(operand, scope), BOOL_TYPE, "Operand of 'not'")
                return BOOL_TYPE
            case ast.UnaryOp(ast.USub(), operand):
                self.require_type(self.type_check_exp(operand, scope), INT_TYPE, "Operand of unary '-'")
                return INT_TYPE
            case ast.BinOp(left, ast.Add() | ast.Sub() | ast.Mult(), right):
                for operand in (left, right):
                    self.require_type(self.type_check_exp(operand, scope), INT_TYPE, "Arithmetic operand")
                return INT_TYPE
            case ast.BoolOp(ast.And() | ast.Or(), values):
                for operand in values:
                    self.require_type(self.type_check_exp(operand, scope), BOOL_TYPE, "Boolean operand")
                return BOOL_TYPE
            case ast.Compare(left, [op], [right]) if isinstance(op, COMPARISON_OPERATORS):
                left_type = self.type_check_exp(left, scope)
                right_type = self.type_check_exp(right, scope)
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    self.require_type(left_type, SCALAR_TYPES, "Left operand of equality comparison")
                    self.require_type(right_type, left_type, "Right operand of equality comparison")
                else:
                    self.require_type(left_type, INT_TYPE, "Left operand of ordering comparison")
                    self.require_type(right_type, INT_TYPE, "Right operand of ordering comparison")
                return BOOL_TYPE
            case ast.IfExp(test, body, orelse):
                self.require_type(self.type_check_exp(test, scope), BOOL_TYPE, "Condition")
                body_type = self.type_check_exp(body, scope)
                else_type = self.type_check_exp(orelse, scope)
                self.require_type(body_type, SCALAR_TYPES, "Then branch of conditional expression")
                self.require_type(else_type, body_type, "Else branch of conditional expression")
                return body_type

            case ast.Call(ast.Name(name), arguments, []):
                return self.type_check_call(name, arguments, scope)
            case _:
                raise TypeError(f"Expected a scalar expression, got {e}")

    def type_check_stmt(self, statement: stmt, scope: TypeScope) -> None:
        match statement:
            case ast.AnnAssign(ast.Name(), _, None):
                pass
            case ast.Assign([ast.Name(name)], value) | ast.AnnAssign(ast.Name(name), _, ast.expr() as value):
                declared = self.variable_type(name, scope)
                self.require_type(self.type_check_exp(value, scope), declared, f"Variable '{name}'")
            case ast.Expr(ast.Call() as call):
                self.type_check_exp(call, scope)
            case ast.Return(ast.expr() as value):
                expected = scope.functions[scope.function_name].result
                self.require_type(self.type_check_exp(value, scope), expected, f"Return from '{scope.function_name}'")
            case ast.If(test, body, orelse) | ast.While(test, body, [] as orelse):
                self.require_type(self.type_check_exp(test, scope), BOOL_TYPE, "Condition")
                self.type_check_stmts(body, scope)
                self.type_check_stmts(orelse, scope)
            case _:
                raise TypeError(f"Unexpected statement during type checking: {statement}")

    def type_check_stmts(self, statements: list[stmt], scope: TypeScope) -> None:
        for statement in statements:
            self.type_check_stmt(statement, scope)

    def type_check_def(self, name: str, body: list[stmt], signatures: dict[str, Signature]) -> None:
        signature = signatures[name]
        variables = signature.parameters.copy()
        for node in ast.walk(Module(body, [])):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                variable = node.target.id
                declared = self.annotation_type(node.annotation, f"variable '{variable}'")
                if variables.setdefault(variable, declared) != declared:
                    raise TypeError(f"Conflicting type annotations for '{variable}'")

        scope = TypeScope(signatures, variables, name)

        self.type_check_stmts(body, scope)

    def type_check(self, p: FunProgram[ast.FunctionDef]) -> None:
        signatures = BUILTINS.copy()
        for definition in p.defs:
            name = definition.name
            parameters = {
                arg.arg: self.annotation_type(arg.annotation, f"parameter '{arg.arg}' in '{name}'")
                for arg in definition.args.args
            }
            result = self.annotation_type(definition.returns, f"return value of '{name}'")
            signatures[name] = Signature(parameters, result)

        for definition in p.defs:
            self.type_check_def(definition.name, definition.body, signatures)

    ############################################################################
    # Shrink
    ############################################################################

    def shrink_exp(self, e: expr) -> expr:
        match e:
            case ast.BoolOp((ast.And() | ast.Or()) as op, values):
                if len(values) < 2:
                    raise ValueError("A boolean operation needs at least two operands")
                result = self.shrink_exp(values[-1])
                for value in reversed(values[:-1]):
                    test = self.shrink_exp(value)
                    if isinstance(op, ast.And):
                        result = ast.IfExp(test, result, ast.Constant(False))
                    else:
                        result = ast.IfExp(test, ast.Constant(True), result)
                return result
            case ast.UnaryOp(ast.Not() | ast.USub(), operand):
                return replace(e, operand=self.shrink_exp(operand))
            case ast.BinOp(left, ast.Add() | ast.Sub() | ast.Mult(), right):
                return replace(e, left=self.shrink_exp(left), right=self.shrink_exp(right))
            case ast.Compare(left, [op], [right]) if isinstance(op, COMPARISON_OPERATORS):
                return replace(e, left=self.shrink_exp(left), comparators=[self.shrink_exp(right)])
            case ast.IfExp(test, body, orelse):
                return replace(
                    e,
                    test=self.shrink_exp(test),
                    body=self.shrink_exp(body),
                    orelse=self.shrink_exp(orelse),
                )
            case ast.Call(ast.Name(), arguments, []):
                return replace(e, args=[self.shrink_exp(arg) for arg in arguments])
            case ast.Name() | ast.Constant():
                return e
            case _:
                raise Exception(f"Unexpected expression during shrink: {e}")

    def shrink_stmt(self, s: stmt) -> stmt | None:
        match s:
            case (
                ast.Expr(ast.Call() as value)
                | ast.Assign([ast.Name()], value)
                | ast.Return(ast.expr() as value)
            ):
                return replace(s, value=self.shrink_exp(value))
            case ast.AnnAssign(ast.Name() as target, _, value):
                if value is None:
                    return None
                return ast.Assign([target], self.shrink_exp(value))
            case ast.While(orelse=orelse) if orelse:
                raise ValueError("while ... else is not supported")
            case ast.If(test, body, orelse) | ast.While(test, body, orelse):
                return replace(
                    s,
                    test=self.shrink_exp(test),
                    body=self.shrink_stmts(body),
                    orelse=self.shrink_stmts(orelse),
                )
            case _:
                raise Exception(f"Unexpected statement during shrink: {s}")

    def shrink_stmts(self, statements: list[stmt]) -> list[stmt]:
        shrunk_statements: list[stmt] = []
        for statement in statements:
            shrunk = self.shrink_stmt(statement)
            if shrunk is not None:
                shrunk_statements.append(shrunk)
        return shrunk_statements

    def shrink_def(self, definition: ast.FunctionDef) -> FunDef:
        params = [arg.arg for arg in definition.args.args]
        return FunDef(definition.name, params, self.shrink_stmts(definition.body))

    def shrink(self, p: FunProgram[ast.FunctionDef]) -> FunProgram:
        return FunProgram([self.shrink_def(definition) for definition in p.defs])

    ############################################################################
    # Remove Complex Operands
    ############################################################################

    def temporary_assignments(self, temporaries: Temporaries) -> list[stmt]:
        return [
            ast.Assign([name], initializer)
            for name, initializer in temporaries
        ]

    def make_begin(self, temporaries: Temporaries, result: expr) -> expr:
        if not temporaries:
            return result

        return Begin(self.temporary_assignments(temporaries), result)

    def rco_exp(self, e: expr, need_atomic: bool) -> tuple[expr, Temporaries]:

        result: expr
        match e:
            case ast.Constant() | ast.Name():
                return e, []

            case ast.Call(ast.Name() as callee, arguments, []):
                atomic_arguments: list[expr] = []
                temporaries: Temporaries = []
                for argument in arguments:
                    atom, argument_temporaries = self.rco_exp(argument, True)
                    atomic_arguments.append(atom)
                    temporaries.extend(argument_temporaries)
                result = ast.Call(callee, atomic_arguments, [])

            case ast.UnaryOp(operator, operand):
                keep_comparison = isinstance(operator, ast.Not) and isinstance(operand, ast.Compare)
                rco_operand, temporaries = self.rco_exp(
                    operand, need_atomic=not keep_comparison,
                )
                result = ast.UnaryOp(operator, rco_operand)

            case ast.BinOp(left, operator, right):
                atomic_left, left_temporaries = self.rco_exp(left, True)
                atomic_right, right_temporaries = self.rco_exp(right, True)
                result = ast.BinOp(atomic_left, operator, atomic_right)
                temporaries = left_temporaries + right_temporaries

            case ast.Compare(left, [operator], [right]):
                atomic_left, left_temporaries = self.rco_exp(left, True)
                atomic_right, right_temporaries = self.rco_exp(right, True)
                result = ast.Compare(atomic_left, [operator], [atomic_right])
                temporaries = left_temporaries + right_temporaries

            case ast.IfExp(test, body, orelse):
                rco_test, temporaries = self.rco_exp(test, False)
                rco_body, body_temporaries = self.rco_exp(body, False)
                rco_orelse, orelse_temporaries = self.rco_exp(orelse, False)
                result = ast.IfExp(
                    rco_test,
                    self.make_begin(body_temporaries, rco_body),
                    self.make_begin(orelse_temporaries, rco_orelse),
                )
            case _:
                raise Exception(f"Unexpected expression during RCO: {e}")

        if not need_atomic:
            return result, temporaries
        temporary = ast.Name(generate_name("tmp"))
        temporaries.append((temporary, result))
        return temporary, temporaries

    def rco_stmt(self, s: stmt) -> list[stmt]:
        match s:
            case (
                ast.Return(ast.expr() as value)
                | ast.Expr(ast.Call() as value)
                | ast.Assign([ast.Name()], value)
            ):
                expression, temporaries = self.rco_exp(value, False)
                return self.temporary_assignments(temporaries) + [replace(s, value=expression)]
            case ast.If(test, body, orelse) | ast.While(test, body, orelse):
                expression, temporaries = self.rco_exp(test, False)
                rco_test = self.make_begin(temporaries, expression)
                rco_body = self.rco_stmts(body)
                rco_orelse = self.rco_stmts(orelse)
                if isinstance(s, ast.If):
                    return [ast.If(rco_test, rco_body, rco_orelse)]
                return [ast.While(rco_test, rco_body, rco_orelse)]
            case _:
                raise Exception(f"Unexpected statement during RCO: {s}")

    def rco_stmts(self, statements: list[stmt]) -> list[stmt]:
        return [
            rco_statement
            for statement in statements
            for rco_statement in self.rco_stmt(statement)
        ]

    def remove_complex_operands(self, p: FunProgram) -> FunProgram:
        return FunProgram([
            FunDef(definition.name, definition.params, self.rco_stmts(definition.body))
            for definition in p.defs
        ])

    ############################################################################
    # ANF Check
    ############################################################################

    def is_atom(self, e: expr) -> bool:
        match e:
            case ast.Name() | ast.Constant(int()):
                return True
            case _:
                return False

    def is_anf_exp(self, e: expr) -> bool:
        match e:
            case ast.BinOp(left, ast.Add() | ast.Sub() | ast.Mult(), right):
                return self.is_atom(left) and self.is_atom(right)
            case ast.UnaryOp(ast.Not(), ast.Compare() as comparison):
                return self.is_anf_exp(comparison)
            case ast.UnaryOp(ast.Not() | ast.USub(), operand):
                return self.is_atom(operand)
            case ast.Compare(left, [op], [right]) if isinstance(op, COMPARISON_OPERATORS):
                return self.is_atom(left) and self.is_atom(right)
            case ast.Call(ast.Name(), arguments, []):
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
                return self.is_atom(e)

    def is_anf_stmt(self, s: stmt) -> bool:
        match s:
            case (
                ast.Return(ast.expr() as value)
                | ast.Expr(ast.Call() as value)
                | ast.Assign([ast.Name()], value)
            ):
                return self.is_anf_exp(value)
            case ast.If(test, body, orelse) | ast.While(test, body, [] as orelse):
                return (
                    self.is_anf_exp(test)
                    and all(self.is_anf_stmt(statement) for statement in body)
                    and all(self.is_anf_stmt(statement) for statement in orelse)
                )
            case _:
                return False

    def is_anf(self, p: FunProgram) -> bool:
        return all(
            self.is_anf_stmt(statement)
            for definition in p.defs
            for statement in definition.body
        )

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
            case ast.Expr(value):
                return self.explicate_effect(value, cont, basic_blocks)
            case ast.Assign([lhs], rhs):
                return self.explicate_assign(rhs, lhs, cont, basic_blocks)
            case ast.Return(ast.expr() as value):
                return self.explicate_tail(value, basic_blocks)
            case ast.If(test, body, orelse):
                goto_cont = self.create_block(cont, basic_blocks)
                new_body   = self.explicate_stmts(body,   goto_cont, basic_blocks)
                new_orelse = self.explicate_stmts(orelse, goto_cont, basic_blocks)
                return self.explicate_pred(test, new_body, new_orelse, basic_blocks)
            case ast.While(test, body, []):
                loop_label = generate_name("loop")
                goto_loop: list[stmt] = [Goto(loop_label)]
                after_loop = self.create_block(cont, basic_blocks)
                new_body = self.explicate_stmts(body, goto_loop, basic_blocks)
                basic_blocks[loop_label] = self.explicate_pred(
                    test, new_body, after_loop, basic_blocks
                )
                return goto_loop
            case _:
                raise Exception(f"Unexpected statement during explicate control: {s}")

    def explicate_tail(self, e: expr, basic_blocks: BasicBlocks) -> list[stmt]:
        match e:
            case IfExp(test, body, orelse):
                then_branch = self.explicate_tail(body, basic_blocks)
                else_branch = self.explicate_tail(orelse, basic_blocks)
                return self.explicate_pred(test, then_branch, else_branch, basic_blocks)
            case Begin() as begin:
                tail = self.explicate_tail(begin.result, basic_blocks)
                return self.explicate_stmts(begin.body, tail, basic_blocks)
            case _:
                return [ast.Return(e)]

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
                | ast.Call(ast.Name(), _, [])
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

    def explicate_control(self, p: FunProgram) -> CFunProgram:
        definitions: list[CFunDef] = []
        for definition in p.defs:
            basic_blocks: BasicBlocks = {}
            basic_blocks[definition.name] = self.explicate_stmts(
                definition.body, [ast.Return(ast.Constant(0))], basic_blocks,
            )
            definitions.append(CFunDef(definition.name, definition.params, basic_blocks))
        return CFunProgram(definitions)

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
        self, target: arg, comparison: ast.cmpop, left: expr, right: expr
    ) -> list[instr]:
        left_arg = self.select_arg(left)
        right_arg = self.select_arg(right)
        match left_arg, right_arg:
            case Immediate(left_value), Immediate(right_value):
                value = self.compare_constants(comparison, left_value, right_value)
                return [Instr("movq", [Immediate(int(value)), target])]
        al = ByteReg("al")
        return [
            Instr("cmpq", [right_arg, left_arg]),
            Instr("set" + self.select_jmp(comparison), [al]),
            Instr("movzbq", [al, target]),
        ]

    def select_call(self, name: str, arguments: list[expr]) -> list[instr]:
        match name:
            case "print":
                label = "print_int"
            case "input_int":
                label = name
            case _:
                label = function_label(name)

        instructions: list[instr] = [
            Instr("movq", [self.select_arg(argument), register])
            for argument, register in zip(arguments, ARGUMENT_REGISTERS)
        ]
        instructions.append(Callq(label, len(arguments)))
        return instructions

    def select_large_return_addend(self, left: arg, addend: int) -> list[instr]:
        # Load the large constant first because the patcher also uses %rax.
        if isinstance(left, Immediate):
            return [Instr("movq", [Immediate(to_signed(left.value + addend)), SCRATCH_REGISTER])]
        return [
            Instr("movq", [Immediate(addend), SCRATCH_REGISTER]),
            Instr("addq", [left, SCRATCH_REGISTER]),
        ]

    def select_assign(self, target: arg, rhs: expr) -> list[instr]:
        match rhs:
            case ast.Constant() | ast.Name():
                return [Instr("movq", [self.select_arg(rhs), target])]

            case ast.Call(ast.Name(name), arguments, []):
                instructions = self.select_call(name, arguments)
                if target != SCRATCH_REGISTER:
                    instructions.append(Instr("movq", [SCRATCH_REGISTER, target]))
                return instructions

            case ast.UnaryOp(ast.Not(), ast.Compare(left, [comparison], [right])):
                return self.select_compare(target, self.negate_comparison(comparison), left, right)

            case ast.Compare(left, [comparison], [right]):
                return self.select_compare(target, comparison, left, right)

            case ast.UnaryOp(ast.Not(), atom):
                return self.select_compare(target, ast.Eq(), atom, ast.Constant(0))

            case ast.UnaryOp(ast.USub(), value):
                return [
                    Instr("movq", [self.select_arg(value), target]),
                    Instr("negq", [target]),
                ]

            case ast.BinOp(left, ast.Add(), right):
                left_arg = self.select_arg(left)
                right_arg = self.select_arg(right)
                if (target == SCRATCH_REGISTER and isinstance(right_arg, Immediate)
                        and self.immediate_size(right_arg.value) == 64):
                    return self.select_large_return_addend(left_arg, right_arg.value)
                if left_arg == target:
                    return [Instr("addq", [right_arg, target])]
                if right_arg == target:
                    return [Instr("addq", [left_arg, target])]
                return [Instr("movq", [left_arg, target]), Instr("addq", [right_arg, target])]

            case ast.BinOp(left, ast.Sub(), right):
                left_arg = self.select_arg(left)
                right_arg = self.select_arg(right)
                if (target == SCRATCH_REGISTER and isinstance(right_arg, Immediate)
                        and self.immediate_size(right_arg.value) == 64):
                    return self.select_large_return_addend(left_arg, to_signed(-right_arg.value))
                if left_arg == target:
                    return [Instr("subq", [right_arg, target])]
                if right_arg == target:
                    return [Instr("negq", [target]), Instr("addq", [left_arg, target])]
                return [Instr("movq", [left_arg, target]), Instr("subq", [right_arg, target])]

            case ast.BinOp(left, ast.Mult(), right):
                left_arg = self.select_arg(left)
                right_arg = self.select_arg(right)
                if isinstance(left_arg, Immediate) and isinstance(right_arg, Immediate):
                    product = left_arg.value * right_arg.value
                    self.immediate_size(product)
                    return [Instr("movq", [Immediate(product), target])]

                if isinstance(right_arg, Immediate):
                    left_arg, right_arg = right_arg, left_arg

                return [
                    Instr("movq", [left_arg, SCRATCH_REGISTER]),
                    Instr("imulq", [right_arg, SCRATCH_REGISTER]),
                    Instr("movq", [SCRATCH_REGISTER, target]),
                ]
            case _:
                raise Exception(f"Unrecognized right-hand side of assignment: {rhs}")

    def select_stmt(self, s: stmt, conclusion: str) -> list[instr]:
        match s:
            case Goto(label):
                return [Jump(label)]

            case ast.Return(ast.expr() as value):
                return self.select_assign(SCRATCH_REGISTER, value) + [Jump(conclusion)]

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

            case ast.Expr(ast.Call(ast.Name(name), arguments, [])):
                return self.select_call(name, arguments)

            case ast.Assign(targets=[ast.Name(lhs)], value=rhs):
                return self.select_assign(Variable(lhs), rhs)
            case _:
                raise Exception(f"Unexpected statement during instruction selection: {s}")

    def select_instructions(self, p: CFunProgram) -> list[X86Def]:
        definitions: list[X86Def] = []
        for definition in p.defs:
            name = function_label(definition.name)
            conclusion = f"{name}.conclusion"
            blocks: dict[str, list[instr]] = {
                (name if label == definition.name else label): [
                    instruction
                    for statement in statements
                    for instruction in self.select_stmt(statement, conclusion)
                ]
                for label, statements in definition.blocks.items()
            }
            # Include the incoming moves in liveness so later arguments stay intact.
            parameter_moves: list[instr] = [
                Instr("movq", [register, Variable(parameter)])
                for parameter, register in zip(definition.params, ARGUMENT_REGISTERS)
            ]
            blocks[name] = parameter_moves + blocks[name]
            definitions.append(X86Def(name, blocks))
        return definitions


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

    def control_flow_graph(self, p: X86Def) -> DirectedAdjList:
        graph = DirectedAdjList()
        for label, instructions in p.blocks.items():
            graph.add_vertex(label)
            for instruction in instructions:
                match instruction:
                    case Jump(label=target) | JumpIf(label=target):
                        graph.add_edge(label, target)
        return graph

    def uncover_live(self, p: X86Def) -> dict[str, list[set[arg]]]:
        cfg = self.control_flow_graph(p)
        predecessors = transpose(cfg)
        worklist = deque(cfg.vertices())
        queued = set(worklist)
        live_before_block: dict[str, set[arg]] = {
            label: set() for label in cfg.vertices()
        }
        live_after: dict[str, list[set[arg]]] = {}
        while worklist:
            label = worklist.pop()
            queued.remove(label)
            instructions = p.blocks.get(label, [])
            live: set[arg] = set()
            after_instructions: list[set[arg]] = [set() for _ in instructions]
            for index in reversed(range(len(instructions))):
                instruction = instructions[index]
                match instruction:
                    case Jump(label=target):
                        live = live_before_block[target].copy()
                    case JumpIf(label=target):
                        live |= live_before_block[target]
                after_instructions[index] = live.copy()
                live = (
                    (live - self.instruction_write(instruction))
                    | self.instruction_read(instruction)
                )
            live_after[label] = after_instructions
            if live != live_before_block[label]:
                live_before_block[label] = live
                for predecessor in predecessors.adjacent(label):
                    if predecessor not in queued:
                        worklist.append(predecessor)
                        queued.add(predecessor)
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
        p: X86Def,
        live_after: dict[str, list[set[arg]]],
    ) -> dict[arg, set[arg]]:
        graph: dict[arg, set[arg]] = {}
        for label, instructions in p.blocks.items():
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
        p: X86Def,
        live_after: dict[str, list[set[arg]]],
        allocatable_registers: tuple[Reg, ...],
        coloring: dict[arg, int],
        homes: dict[Variable, arg],
    ) -> None:
        print(
            "Available for allocation: "
            + ", ".join(map(str, allocatable_registers)),
            file=sys.stderr,
        )

        rows: list[tuple[str, str, str]] = []
        for label, instructions in p.blocks.items():
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
        p: X86Def,
        registers_cap: int = AVAILABLE_REGISTER_COUNT,
        debug_allocate_registers: bool = False,
    ) -> X86Def:
        if registers_cap < 1:
            raise ValueError("registers_cap must be at least 1")
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
        blocks: dict[str, list[instr]] = {
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
            for label, instructions in p.blocks.items()
        }
        return X86Def(p.name, blocks)

    ############################################################################
    # Patch Instructions
    ############################################################################
    def patch_instr(self, i: instr) -> list[instr]:
        # The scratch register is reserved by the allocator.
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
                        assert destination != SCRATCH_REGISTER, "Patching would overwrite %rax"
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

    def patch_instructions(self, p: X86Def) -> X86Def:
        blocks: dict[str, list[instr]] = {}
        for label, instructions in p.blocks.items():
            blocks[label] = [
                patched
                for instruction in instructions
                for patched in self.patch_instr(instruction)
            ]
        return X86Def(p.name, blocks)

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

    def add_function_frame(self, p: X86Def) -> dict[str, list[instr]]:
        instructions = [i for block in p.blocks.values() for i in block]
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

        return {
            **p.blocks,
            p.name: prologue + p.blocks[p.name],
            f"{p.name}.conclusion": conclusion,
        }

    def prelude_and_conclusion(self, definitions: list[X86Def]) -> X86Program:
        blocks: dict[str, list[instr]] = {}
        for definition in definitions:
            blocks.update(self.add_function_frame(definition))
        return X86Program(blocks, frozenset(d.name for d in definitions))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--no-type-check", action="store_true", help="skip source type checking")
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
    program = compiler.split_module(module)
    compiler.check_functions(program)
    if not args.no_type_check:
        compiler.type_check(program)
    checkpoint(args, "type_check", module)
    shrinked = compiler.shrink(program)
    checkpoint(args, "shrink", shrinked)
    simplified = compiler.remove_complex_operands(shrinked)
    assert compiler.is_anf(simplified), \
        f"The AST is not in A-normal form:\n{simplified}"
    checkpoint(args, "remove_complex_operands", simplified)
    explicated = compiler.explicate_control(simplified)
    checkpoint(args, "explicate_control", explicated)
    instruction_list = compiler.select_instructions(explicated)
    checkpoint(args, "select_instructions", instruction_list)
    allocated = [
        compiler.allocate_registers(definition, args.registers_cap, args.debug_allocate_registers)
        for definition in instruction_list
    ]
    checkpoint(args, "allocate_registers", allocated)
    patched_instructions = [compiler.patch_instructions(definition) for definition in allocated]
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

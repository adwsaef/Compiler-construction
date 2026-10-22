# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
import ast
import argparse

from ast import (Constant, expr, stmt, Module, Name)
from ast_utils import generate_name  # type: ignore[import-untyped]
from x86_ast import (X86Program, instr, arg, Variable, Immediate, Instr, Callq, Reg, Deref, is_memory_arg) # type: ignore[import-untyped]
from x86_interp import checkpoint, add_arguments, interp_x86 # type: ignore[import-untyped]

Binding = tuple[Name, expr]
Temporaries = list[Binding]


class Compiler:


    ############################################################################
    # Remove Complex Operands
    ############################################################################

    def rco_exp(self, e: expr, need_atomic : bool) -> tuple[expr, Temporaries]:
        match e:
            case ast.Constant() | ast.Name():
                return e, []

            case ast.Call(ast.Name("input_int"), []):
                if not need_atomic:
                    return e, []

                temporary = ast.Name(generate_name("tmp"))
                return temporary, [(temporary, e)]

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
            case _:
                raise Exception(f"Unexpected statement during RCO: {s}")

    def remove_complex_operands(self, p: Module) -> Module:
        statements: list[stmt] = []
        for statement in p.body:
            statements.extend(self.rco_stmt(statement))
        return ast.Module(statements)

    ############################################################################
    # Select Instructions
    ############################################################################

    # The expression e passed to select_arg should furthermore be an atom.
    # (But there is no type for atoms, so the type of e is given as expr.)
    def select_arg(self, e: expr) -> arg:
        match e:
            case Constant(value):
                match value:
                    case int():
                        return Immediate(value)
                    case _:
                        raise Exception(f"Only integers allowed {e}")
            case Name(id):
                return Variable(id)
            case _:
                raise Exception(f"Unexpected expression {e}")

    def select_stmt(self, s: stmt) -> list[instr]:
        match s:
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
                            return [
                                Instr("movq", [Immediate(left_arg.value * right_arg.value), Variable(lhs)])
                            ]

                        if isinstance(right_arg, Immediate):
                            left_arg, right_arg = right_arg, left_arg

                        # Right arg is not immadiete, emitted imulq is correct.
                        return [
                            Instr("movq", [left_arg, Reg("rax")]),
                            Instr("imulq", [right_arg, Reg("rax")]),
                            Instr("movq", [Reg("rax"), Variable(lhs)]),
                        ]
                    case _:
                        raise Exception(f"Unrecognized right-hand side of assignment: {rhs}")
            case _:
                raise Exception(f"Unexpected statement {s}, it must be assignment or print")

    def select_instructions(self, p: Module) -> X86Program:
        instructions: list[instr] = []
        for s in p.body:
            instructions.extend(self.select_stmt(s))
        return X86Program(instructions)


    ############################################################################
    # Assign Homes
    ############################################################################

    def verify_variables_initialized(self, p: X86Program) -> None:
        initialized: set[Variable] = set()

        for i in p.body:
            if type(i) is not Instr:
                continue

            reads = i.args
            writes: tuple[arg, ...] = ()
            if i.instr == "movq":
                reads = i.args[:1]
                writes = i.args[1:]

            for a in reads:
                if type(a) is Variable and a not in initialized:
                    raise Exception(f"Variable {a} used before assignment")

            initialized.update(
                a for a in writes
                if type(a) is Variable
            )

    def assign_homes_arg(self, a: arg, home: dict[Variable, arg]) -> arg:
        match a:
            case Variable():
                if a not in home:
                    home[a] = Deref("rbp", -8 * (len(home) + 1))
                return home[a]
            case _:
                return a

    def assign_homes_instr(self, i: instr, home: dict[Variable, arg]) -> instr:
        match i:
            case Instr():
                return Instr(
                    i.instr,
                    [self.assign_homes_arg(a, home) for a in i.args],
                )
            case _:
                return i

    def assign_homes(self, p: X86Program) -> X86Program:
        assert type(p.body) is list
        self.verify_variables_initialized(p)
        home: dict[Variable, arg] = {}
        return X86Program([self.assign_homes_instr(i, home) for i in p.body])

    ############################################################################
    # Patch Instructions
    ############################################################################
    def immediate_size(self, value: int) -> int:
        if -(2 ** 31) <= value < 2 ** 31:
            return 32
        if -(2 ** 63) <= value < 2 ** 63:
            return 64
        raise Exception(f"Immediate value {value} does not fit in 64 bits")

    def patch_instr(self, i: instr) -> list[instr]:
        # Rax is only used for after call
        match i:
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
                        scratch = Reg("rax")
                        return [
                            Instr("movq", [Immediate(value), scratch]),
                            Instr(i.instr, [scratch, destination]),
                        ]

                    case [source, destination] \
                            if is_memory_arg(source) and is_memory_arg(destination):
                        return [
                            Instr("movq", [source, Reg("rax")]),
                            Instr(i.instr, [Reg("rax"), destination]),
                        ]
        return [i]

    def patch_instructions(self, p: X86Program) -> X86Program:
        instructions: list[instr] = []
        for i in p.body:
            match i:
                case instr():
                    instructions.extend(self.patch_instr(i))
        return X86Program(instructions)     

    ############################################################################
    # Prelude & Conclusion
    ############################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        stack_space = 0
        for i in p.body:
            match i:
                case Instr():
                    for a in i.args:
                        match a:
                            case Deref() if a.reg == "rbp" and a.offset < 0:
                                stack_space = max(stack_space, -a.offset)
        stack_space = (stack_space + 15) // 16 * 16

        prologue = [
            Instr("pushq", [Reg("rbp")]),
            Instr("movq", [Reg("rsp"), Reg("rbp")]),
            Instr("subq", [Immediate(stack_space), Reg("rsp")]),
        ]
        conclusion = [
            Instr("movq", [Reg("rbp"), Reg("rsp")]),
            Instr("movq", [Immediate(0), Reg("rax")]),
            Instr("popq", [Reg("rbp")]),
            Instr("retq", []),
        ]

        assert type(p.body) is list, (
            "prelude_and_conclusion expects a flat instruction list"
        )

        p.body = prologue + p.body + conclusion

        return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.input, "r") as file:
        module = ast.parse(file.read())

    compiler = Compiler()

    simplified = compiler.remove_complex_operands(module)
    checkpoint(args, "remove_complex_operands", simplified)
    instruction_list = compiler.select_instructions(simplified)
    checkpoint(args, "select_instructions", instruction_list)
    assigned_homes = compiler.assign_homes(instruction_list)
    checkpoint(args, "assign_homes", assigned_homes)
    patched_instructions = compiler.patch_instructions(assigned_homes)
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

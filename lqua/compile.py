# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
import ast
import argparse
from ast import (Constant, expr, stmt, Module, Name)
from x86_ast import (X86Program, instr, arg, Variable, Immediate, Instr, Callq, Reg, Deref, is_memory_arg)
from x86_interp import checkpoint, add_arguments, interp_x86
import os

Binding = tuple[Name, expr]
Temporaries = list[Binding]


class Compiler:


    ############################################################################
    # Remove Complex Operands
    ############################################################################

    def rco_exp(self, e: expr, need_atomic : bool) -> tuple[expr, Temporaries]:
        # YOUR CODE HERE
        pass        

    def rco_stmt(self, s: stmt) -> list[stmt]:
        # YOUR CODE HERE
        pass        

    def remove_complex_operands(self, p: Module) -> Module:
        # YOUR CODE HERE
        pass                

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

            case ast.Assign(targets=[ast.Name(name)], value=right):
                match right:
                    case ast.Constant() | ast.Name():
                        return [
                            Instr("movq", [self.select_arg(right), Variable(name)]),
                            # Instr("movq", [self.select_arg(right), Reg("rax")]),
                            #Instr("movq", [Reg("rax"), Variable(name)])
                        ]

                    case ast.Call(ast.Name("input_int"), []):
                        return [
                            Callq("input_int", 0),
                            Instr("movq", [Reg("rax"), Variable(name)]),
                        ]

                    case ast.UnaryOp(ast.USub(), value):
                        return [
                            Instr("movq", [self.select_arg(value), Reg("rax")]),
                            Instr("negq", [Reg("rax")]),
                            Instr("movq", [Reg("rax"), Variable(name)])
                        ]

                    case ast.BinOp(left, ast.Add(), right):
                        return [
                            Instr("movq", [self.select_arg(left), Reg("rax")]),
                            Instr("addq", [self.select_arg(right), Reg("rax")]),
                            Instr("movq", [Reg("rax"), Variable(name)]),
                        ]
                    case ast.BinOp(left, ast.Sub(), right):
                        return [
                            Instr("movq", [self.select_arg(left), Reg("rax")]),
                            Instr("subq", [self.select_arg(right), Reg("rax")]),
                            Instr("movq", [Reg("rax"), Variable(name)]),
                        ]

                    case ast.BinOp(left, ast.Mult(), right):
                        return [
                            Instr("movq", [self.select_arg(left), Reg("rax")]),
                            Instr("imulq", [self.select_arg(right), Reg("rax")]),
                            Instr("movq", [Reg("rax"), Variable(name)]),
                        ]
                    case _:
                        raise Exception(f"Unrecognized right-hand side of assignment: {right}")
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

    def assign_homes_arg(self, a: arg, home: dict[Variable, arg]) -> arg:
        match a:
            case Variable():
                return home[a]
            case _:
                return a

    def assign_homes_instr(self, i: instr,
                           home: dict[Variable, arg]) -> instr:
        match i:
            case Instr(name, args):
                return Instr(name, [self.assign_homes_arg(a, home) for a in args])
            case _:
                return i

    def assign_homes(self, p: X86Program) -> X86Program:
        home: dict[Variable, arg] = {}
        for i in p.body:
            match i:
                case Instr(_, args):
                    for a in args:
                        match a:
                            case Variable():
                                if a not in home:
                                    home[a] = Deref("rbp", -8 * (len(home) + 1))

        instructions: list[instr] = []
        for i in p.body:
            match i:
                case instr():
                    instructions.append(self.assign_homes_instr(i, home))

        return X86Program(instructions)

    ############################################################################
    # Patch Instructions
    ############################################################################
    def patch_instr(self, i: instr) -> list[instr]:
        match i:
            case Instr("movq", [Immediate(value), destination]) \
                    if value >= 2**31 or value < -2**31:
                return [
                    Instr("movq", [Immediate(value), Reg("rax")]),
                    Instr("movq", [Reg("rax"), destination]),
                ]

            case Instr(operation, [source, destination]) \
                    if is_memory_arg(source) and is_memory_arg(destination):
                return [
                    Instr("movq", [source, Reg("rax")]),
                    Instr(operation, [Reg("rax"), destination]),
                ]

            case _:
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
                case Instr(_, args):
                    for a in args:
                        match a:
                            case Deref("rbp", offset) if offset < 0:
                                stack_space = max(stack_space, -offset)

        prologue = [
            Instr("pushq", [Reg("rbp")]),
            Instr("movq", [Reg("rsp"), Reg("rbp")]),
            Instr("subq", [Immediate(stack_space), Reg("rsp")]),
        ]
        conclusion = [
            Instr("addq", [Immediate(stack_space), Reg("rsp")]),
            Instr("popq", [Reg("rbp")]),
            Instr("retq", []),
        ]

        instructions: list[instr] = []
        for instruction in prologue:
            instructions.append(instruction)

        for instruction in p.body:
            match instruction:
                case instr():
                    instructions.append(instruction)
                case _:
                    raise TypeError(
                        "prelude_and_conclusion expects a flat instruction list"
                    )

        for instruction in conclusion:
            instructions.append(instruction)

        return X86Program(instructions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.input, "r") as file:
        module = ast.parse(file.read())

    compiler = Compiler()

    instruction_list = compiler.select_instructions(module)
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

    with open(args.output, "w") as output_file:
        output_file.write(str(added_prelude_conclusion))

if __name__ == "__main__":
    main()

# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
import ast
import argparse
import sys
from collections.abc import Sequence

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
    # Register Allocation
    ############################################################################

    def uncover_live(self, p: X86Program) -> dict[int, set[arg]]:
        assert isinstance(p.body, list)
        argument_registers: tuple[arg, ...] = tuple(
            Reg(name) for name in ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
        )
        caller_saved: set[arg] = set(argument_registers)
        caller_saved.update((Reg("rax"), Reg("r10"), Reg("r11")))

        def locations(*operands: arg) -> set[arg]:
            return {
                operand for operand in operands
                if isinstance(operand, (Variable, Reg, Deref))
            }

        def read_write(i: instr) -> tuple[set[arg], set[arg]]:
            match i:
                case Callq(num_args=arity):
                    return set(argument_registers[:arity]), caller_saved
                case Instr("movq", [source, destination]):
                    return locations(source), locations(destination)
                case Instr("addq" | "subq" | "imulq", [source, destination]):
                    return locations(source, destination), locations(destination)
                case Instr("negq", [destination]):
                    return locations(destination), locations(destination)
                case _:
                    raise ValueError(f"Liveness is not defined for {i}")

        live: set[arg] = set()
        live_after: dict[int, set[arg]] = {}
        for index in reversed(range(len(p.body))):
            live_after[index] = live.copy()
            reads, writes = read_write(p.body[index])
            live = (live - writes) | reads
        return live_after

    def debug_print_live(
        self, live_instructions: dict[int, tuple[instr, set[arg]]]
    ) -> None:
        rows: list[tuple[str, str, str]] = []
        for index in sorted(live_instructions):
            instruction, live_after = live_instructions[index]
            _, edges = self.instruction_interference(instruction, live_after)
            instruction_text = str(instruction).strip()
            live_text = "{" + ", ".join(sorted(map(str, live_after))) + "}"
            edge_text = ", ".join(
                f"{destination}-{neighbor}"
                for destination, neighbor in sorted(
                    edges, key=lambda edge: (str(edge[0]), str(edge[1]))
                )
            ) or "--"
            rows.append((instruction_text, live_text, edge_text))

        instruction_width = max((len(row[0]) for row in rows), default=0)
        instruction_width = max(instruction_width, len("Instruction"))
        live_width = max((len(row[1]) for row in rows), default=0)
        live_width = max(live_width, len("live-after"))
        print(
            f"{'Instruction':<{instruction_width}}  "
            f"{'live-after':<{live_width}}  edges added",
            file=sys.stderr,
        )
        for instruction, live, edges in rows:
            print(
                f"{instruction:<{instruction_width}}  "
                f"{live:<{live_width}}  {edges}",
                file=sys.stderr,
            )

    def instruction_interference(
        self, instruction: instr, live_after: set[arg]
    ) -> tuple[set[arg], set[tuple[arg, arg]]]:
        move_source: arg | None = None
        match instruction:
            case Callq():
                writes: set[arg] = {
                    Reg(name) for name in
                    ("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11")
                }
            case Instr("movq", [source, destination]):
                writes = {destination}
                move_source = source
            case Instr("addq" | "subq" | "imulq", [_, destination]):
                writes = {destination}
            case Instr("negq", [destination]):
                writes = {destination}
            case _:
                raise ValueError(f"Interference is not defined for {instruction}")

        edges: set[tuple[arg, arg]] = {
            (destination, neighbor)
            for destination in writes if isinstance(destination, (Variable, Reg))
            for neighbor in live_after if isinstance(neighbor, (Variable, Reg))
            and neighbor != destination and neighbor != move_source
        }
        return writes, edges

    def build_interface_graph(
        self, live_instructions: dict[int, tuple[instr, set[arg]]]
    ) -> dict[arg, set[arg]]:
        """Connect each written location to the locations live after it."""
        graph: dict[arg, set[arg]] = {}
        for index in sorted(live_instructions):
            instruction, live_after = live_instructions[index]
            writes, edges = self.instruction_interference(instruction, live_after)

            for location in live_after | writes:
                if isinstance(location, (Variable, Reg)):
                    graph.setdefault(location, set())

            for destination, neighbor in edges:
                graph[destination].add(neighbor)
                graph[neighbor].add(destination)
        return graph

    def color_graph(
        self, graph: dict[arg, set[arg]], colors: Sequence[int],
        precolored: dict[arg, int]
    ) -> dict[arg, int]:
        """Color a graph by highest saturation, using the first free color."""
        color = precolored.copy()
        uncolored = set(graph) - color.keys()
        while uncolored:
            vertex = max(
                uncolored,
                key=lambda v: (
                    len({color[n] for n in graph[v] if n in color}),
                    len(graph[v]),
                    str(v),
                ),
            )
            used = {color[n] for n in graph[vertex] if n in color}
            available = next((c for c in colors if c not in used), None)
            if available is None:
                raise ValueError(f"No color available for {vertex}")
            color[vertex] = available
            uncolored.remove(vertex)
        return color

    def allocate_registers(self, p: X86Program, debug: bool = False) -> X86Program:
        live_after = self.uncover_live(p)
        assert isinstance(p.body, list)
        live_instructions = {
            index: (instruction, live_after[index])
            for index, instruction in enumerate(p.body)
        }
        graph = self.build_interface_graph(live_instructions)
        physical_registers = tuple(map(Reg, (
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9",
            "r10", "r11", "r12", "r13", "r14", "r15",
        )))
        precolored: dict[arg, int] = {
            register: index for index, register in enumerate(physical_registers)
            if register in graph
        }
        self.color_graph(graph, range(len(physical_registers) + len(graph)), precolored)
        if debug:
            self.debug_print_live(live_instructions)
        return p

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
                        return [
                            Instr("movq", [Immediate(value),  Reg("rax")]),
                            Instr(i.instr, [Reg("rax"), destination]),
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
    parser.add_argument("--debug", action="store_true", help="print instructions and live-after sets")
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.input, "r") as file:
        module = ast.parse(file.read())

    compiler = Compiler()

    simplified = compiler.remove_complex_operands(module)
    checkpoint(args, "remove_complex_operands", simplified)
    instruction_list = compiler.select_instructions(simplified)
    checkpoint(args, "select_instructions", instruction_list)
    allocated = compiler.allocate_registers(instruction_list, debug=args.debug)
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

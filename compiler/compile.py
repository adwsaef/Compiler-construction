# Adapted by permission from https://github.com/IUCompilerCourse/python-student-support-code
# See EOC-LICENSE for details
import ast
import argparse
import sys

from ast import (Constant, expr, stmt, Module, Name)
from ast_utils import generate_name  # type: ignore[import-untyped]
from x86_ast import (X86Program, instr, arg, Variable, Immediate, Instr, Callq, Reg, Deref, is_memory_arg) # type: ignore[import-untyped]
from x86_interp import checkpoint, add_arguments, interp_x86 # type: ignore[import-untyped]

Binding = tuple[Name, expr]
Temporaries = list[Binding]

SCRATCH_REGISTER = Reg("rax")
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
                            Instr("movq", [left_arg, SCRATCH_REGISTER]),
                            Instr("imulq", [right_arg, SCRATCH_REGISTER]),
                            Instr("movq", [SCRATCH_REGISTER, Variable(lhs)]),
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

    def operand_locations(self, operand: arg) -> set[arg]:
        match operand:
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
            case Instr(instr="movq", args=[source, _]):
                return self.operand_locations(source)
            case Instr(
                instr="addq" | "subq" | "imulq",
                args=[source, destination],
            ):
                return self.operand_locations(source) | self.operand_locations(destination)
            case Instr(instr="negq", args=[destination]):
                return self.operand_locations(destination)
            case _:
                raise ValueError(f"Reads are not defined for {instruction}")

    def instruction_write(self, instruction: instr) -> set[arg]:
        match instruction:
            case Callq():
                return set(CALLER_SAVED_REGISTERS)
            case Instr(
                instr="movq" | "addq" | "subq" | "imulq",
                args=[_, destination],
            ) | Instr(instr="negq", args=[destination]):
                return self.operand_locations(destination)
            case _:
                raise ValueError(f"Writes are not defined for {instruction}")

    def uncover_live(self, p: X86Program) -> dict[int, set[arg]]:
        assert isinstance(p.body, list)
        live: set[arg] = set()
        live_after: dict[int, set[arg]] = {}
        for index in reversed(range(len(p.body))):
            live_after[index] = live.copy()
            reads = self.instruction_read(p.body[index])
            writes = self.instruction_write(p.body[index])
            live = (live - writes) | reads
        return live_after

    def instruction_edges(
        self,
        instruction: instr,
        live_after: set[arg],
        writes: set[arg],
    ) -> set[tuple[arg, arg]]:
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

    def build_interface_graph(
        self,
        p: X86Program,
        live_after: dict[int, set[arg]],
    ) -> dict[arg, set[arg]]:
        assert isinstance(p.body, list)
        graph: dict[arg, set[arg]] = {}
        for index, instruction in enumerate(p.body):
            writes = self.instruction_write(instruction)
            for location in live_after[index] | writes:
                graph.setdefault(location, set())

            edges = self.instruction_edges(instruction, live_after[index], writes)
            for destination, neighbor in edges:
                graph[destination].add(neighbor)
                graph[neighbor].add(destination)
        return graph


    def debug_print_live(
        self,
        p: X86Program,
        live_after: dict[int, set[arg]],
        allocatable_registers: tuple[Reg, ...],
    ) -> None:
        assert isinstance(p.body, list)
        print(
            "Available for allocation: "
            + ", ".join(map(str, allocatable_registers)),
            file=sys.stderr,
        )
        rows: list[tuple[str, str, str]] = []
        for index, instruction in enumerate(p.body):
            instruction_text = str(instruction).strip()
            live_text = "{" + ", ".join(sorted(map(str, live_after[index]))) + "}"
            edges = self.instruction_edges(
                instruction, live_after[index], self.instruction_write(instruction)
            )
            edge_text = ", ".join(
                f"{destination}-{neighbor}"
                for destination, neighbor in sorted(
                    edges,
                    key=lambda edge: (str(edge[0]), str(edge[1])),
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
        for instruction_text, live_text, edge_text in rows:
            print(
                f"{instruction_text:<{instruction_width}}  "
                f"{live_text:<{live_width}}  {edge_text}",
                file=sys.stderr,
            )

    def check_variables_defined(self, instructions: list[instr]) -> None:
        defined: set[arg] = set()
        for instruction in instructions:
            unassigned = {
                location for location in self.instruction_read(instruction)
                if isinstance(location, Variable) and location not in defined
            }
            if unassigned:
                names = ", ".join(sorted(str(location) for location in unassigned))
                raise ValueError(f"Variable read before assignment: {names}")
            defined.update(
                location for location in self.instruction_write(instruction)
                if isinstance(location, Variable)
            )

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
        debug: bool = False,
    ) -> X86Program:
        if registers_cap < 1:
            raise ValueError("registers_cap must be at least 1")
        assert isinstance(p.body, list)
        self.check_variables_defined(p.body)
        live_after = self.uncover_live(p)
        graph = self.build_interface_graph(p, live_after)

        registers_cap = min(registers_cap, AVAILABLE_REGISTER_COUNT)
        coloring: dict[arg, int] = {
            register: REGISTER_TO_COLOR[register]
            for register in graph if isinstance(register, Reg)
        }
        self.color_graph(graph, coloring)

        stack_offsets: dict[int, int] = {}
        homes: dict[Variable, arg] = {}
        for location, color in coloring.items():
            if not isinstance(location, Variable):
                continue
            if color < registers_cap:
                homes[location] = COLOR_TO_REGISTER[color]
            else:
                if color not in stack_offsets:
                    slot = len(stack_offsets) + 1
                    stack_offsets[color] = -8 * (len(CALLEE_SAVED_REGISTERS) + slot)
                homes[location] = Deref("rbp", stack_offsets[color])

        if debug:
            self.debug_print_live(
                p, live_after,
                tuple(COLOR_TO_REGISTER[color] for color in range(registers_cap)),
            )
        p.body = [
            Instr(
                instruction.instr,
                [
                    homes[operand] if isinstance(operand, Variable) else operand
                    for operand in instruction.args
                ],
            )
            if isinstance(instruction, Instr) else instruction
            for instruction in p.body
        ]
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
        # Keep the scratch register out of the allocatable register set.
        match i:
            case Instr(instr="movq", args=[source, destination]) if source == destination:
                return []
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
        assert isinstance(p.body, list)
        stack_space = 0
        used_callee_saved: set[Reg] = set()
        for instruction in p.body:
            if not isinstance(instruction, Instr):
                continue
            for operand in instruction.args:
                if isinstance(operand, Deref) and operand.reg == "rbp":
                    stack_space = max(stack_space, -operand.offset)
                elif isinstance(operand, Reg) and operand in CALLEE_SAVED_REGISTERS:
                    used_callee_saved.add(operand)

        saved_registers: list[tuple[Reg, Deref]] = []
        for index, register in enumerate(CALLEE_SAVED_REGISTERS):
            if register in used_callee_saved:
                slot = Deref("rbp", -8 * (index + 1))
                saved_registers.append((register, slot))
                stack_space = max(stack_space, -slot.offset)
        stack_space = (stack_space + 15) // 16 * 16

        prologue = [
            Instr("pushq", [Reg("rbp")]),
            Instr("movq", [Reg("rsp"), Reg("rbp")]),
            Instr("subq", [Immediate(stack_space), Reg("rsp")]),
        ] + [Instr("movq", [register, slot]) for register, slot in saved_registers]
        conclusion = [
            Instr("movq", [slot, register])
            for register, slot in reversed(saved_registers)
        ] + [
            Instr("movq", [Reg("rbp"), Reg("rsp")]),
            Instr("movq", [Immediate(0), Reg("rax")]),
            Instr("popq", [Reg("rbp")]),
            Instr("retq", []),
        ]

        p.body = prologue + p.body + conclusion

        return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--debug", action="store_true", help="print instructions and live-after sets")
    parser.add_argument(
        "--registers-cap", type=int, default=AVAILABLE_REGISTER_COUNT,
        help=f"number of allocatable registers (at least 1; {AVAILABLE_REGISTER_COUNT} available)",
    )
    add_arguments(parser)
    args = parser.parse_args()
    if args.registers_cap < 1:
        parser.error("--registers-cap must be at least 1")

    with open(args.input, "r") as file:
        module = ast.parse(file.read())

    compiler = Compiler()

    simplified = compiler.remove_complex_operands(module)
    checkpoint(args, "remove_complex_operands", simplified)
    instruction_list = compiler.select_instructions(simplified)
    checkpoint(args, "select_instructions", instruction_list)
    allocated = compiler.allocate_registers(
        instruction_list, registers_cap=args.registers_cap, debug=args.debug
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

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
RESERVED_REGISTERS = (
    Reg("rax"), Reg("rsp"), Reg("rbp"), Reg("r11"), Reg("r15"),
)
AVAILABLE_REGISTERS = tuple(
    register for register in CALLEE_SAVED_REGISTERS + CALLER_SAVED_REGISTERS
    if register not in RESERVED_REGISTERS
)


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

    def instruction_read(self, instruction: instr) -> set[arg]:
        match instruction:
            case Callq(num_args=arity):
                return set(ARGUMENT_REGISTERS[:arity])
            case Instr(instr="movq", args=[source, _]):
                return {source} if isinstance(source, (Variable, Reg, Deref)) else set()
            case Instr(
                instr="addq" | "subq" | "imulq",
                args=[source, destination],
            ):
                return {
                    operand for operand in (source, destination)
                    if isinstance(operand, (Variable, Reg, Deref))
                }
            case Instr(instr="negq", args=[destination]):
                return {destination} if isinstance(destination, (Variable, Reg, Deref)) else set()
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
                return {destination} if isinstance(destination, (Variable, Reg, Deref)) else set()
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

    def debug_print_live(
        self,
        p: X86Program,
        live_after: dict[int, set[arg]],
        edges_by_index: dict[int, set[tuple[arg, arg]]],
    ) -> None:
        assert isinstance(p.body, list)
        rows: list[tuple[str, str, str]] = []
        for index, instruction in enumerate(p.body):
            instruction_text = str(instruction).strip()
            live_text = "{" + ", ".join(sorted(map(str, live_after[index]))) + "}"
            edge_text = ", ".join(
                f"{destination}-{neighbor}"
                for destination, neighbor in sorted(
                    edges_by_index[index],
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

    def build_interface_graph(
        self,
        p: X86Program,
        live_after: dict[int, set[arg]],
    ) -> tuple[dict[arg, set[arg]], dict[int, set[tuple[arg, arg]]]]:
        """Connect each written location to the locations live after it."""
        assert isinstance(p.body, list)
        writes_by_index = [self.instruction_write(i) for i in p.body]
        locations: set[arg] = set()
        for after in live_after.values():
            locations.update(after)
        for writes in writes_by_index:
            locations.update(writes)
        for instruction in p.body:
            if isinstance(instruction, Instr):
                locations.update(
                    operand for operand in instruction.args
                    if isinstance(operand, Variable)
                )
        graph: dict[arg, set[arg]] = {
            location: set()
            for location in locations
            if isinstance(location, (Variable, Reg))
        }
        edges_by_index: dict[int, set[tuple[arg, arg]]] = {}
        for index, instruction in enumerate(p.body):
            writes = writes_by_index[index]

            match instruction:
                case Instr(instr="movq", args=[source, _]):
                    neighbors = live_after[index] - {source}
                case _:
                    neighbors = live_after[index]
            edges: set[tuple[arg, arg]] = {
                (destination, neighbor)
                for destination in writes if isinstance(destination, (Variable, Reg))
                for neighbor in neighbors if isinstance(neighbor, (Variable, Reg))
                and neighbor != destination
            }
            for destination, neighbor in edges:
                graph[destination].add(neighbor)
                graph[neighbor].add(destination)
            edges_by_index[index] = edges
        return graph, edges_by_index

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

    def allocate_registers(
        self,
        p: X86Program,
        registers_cap: int = len(AVAILABLE_REGISTERS),
        debug: bool = False,
    ) -> X86Program:
        """Replace variables with up to registers_cap registers or stack slots."""
        if not 0 <= registers_cap <= len(AVAILABLE_REGISTERS):
            raise ValueError(
                f"registers_cap must be between 0 and {len(AVAILABLE_REGISTERS)}"
            )
        assert isinstance(p.body, list)
        defined: set[arg] = set()
        for instruction in p.body:
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
        live_after = self.uncover_live(p)
        graph, edges_by_index = self.build_interface_graph(p, live_after)

        fixed_registers = RESERVED_REGISTERS + AVAILABLE_REGISTERS
        register_colors = {
            register: index for index, register in enumerate(fixed_registers)
        }
        precolored: dict[arg, int] = {
            register: register_colors[register]
            for register in graph if isinstance(register, Reg)
        }
        spill_start = len(fixed_registers)
        colors = [
            register_colors[register]
            for register in AVAILABLE_REGISTERS[:registers_cap]
        ] + list(range(spill_start, spill_start + len(graph)))
        coloring = self.color_graph(graph, colors, precolored)
        spill_colors = sorted({
            coloring[location] for location in graph
            if isinstance(location, Variable) and coloring[location] >= spill_start
        })
        spill_homes = {
            color: Deref("rbp", -8 * (len(CALLEE_SAVED_REGISTERS) + slot))
            for slot, color in enumerate(spill_colors, start=1)
        }
        homes: dict[Variable, arg] = {
            location: (
                fixed_registers[coloring[location]]
                if coloring[location] < spill_start
                else spill_homes[coloring[location]]
            )
            for location in graph if isinstance(location, Variable)
        }
        if debug:
            self.debug_print_live(p, live_after, edges_by_index)
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
        for i in p.body:
            match i:
                case Instr():
                    for a in i.args:
                        match a:
                            case Deref() if a.reg == "rbp" and a.offset < 0:
                                stack_space = max(stack_space, -a.offset)
        saved_registers = [
            (register, Deref("rbp", -8 * (index + 1)))
            for index, register in enumerate(CALLEE_SAVED_REGISTERS)
            if any(
                isinstance(instruction, Instr) and register in instruction.args
                for instruction in p.body
            )
        ]
        for _, slot in saved_registers:
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
        "--registers-cap", type=int, default=len(AVAILABLE_REGISTERS),
        help=f"maximum allocatable registers (0-{len(AVAILABLE_REGISTERS)})",
    )
    add_arguments(parser)
    args = parser.parse_args()
    if not 0 <= args.registers_cap <= len(AVAILABLE_REGISTERS):
        parser.error(f"--registers-cap must be between 0 and {len(AVAILABLE_REGISTERS)}")

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

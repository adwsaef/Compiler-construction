from contextlib import redirect_stderr
from io import StringIO
import unittest

from compiler.compile import Compiler
from x86_ast import Callq, Immediate, Instr, Reg, Variable, X86Program


class UncoverLiveTests(unittest.TestCase):
    def graph_for(self, instructions):
        compiler = Compiler()
        live_after = compiler.uncover_live(X86Program(instructions))
        return compiler.build_interface_graph({
            index: (instruction, live_after[index])
            for index, instruction in enumerate(instructions)
        })

    def test_live_after_running_example(self):
        v, w, x, y, z = map(Variable, ("v", "w", "x", "y", "z"))
        tmp1, tmp2 = Variable("_1"), Variable("_2")
        instructions = [
            Instr("movq", [Immediate(1), v]),
            Instr("movq", [Immediate(42), w]),
            Instr("movq", [v, x]),
            Instr("addq", [Immediate(7), x]),
            Instr("movq", [x, y]),
            Instr("movq", [x, z]),
            Instr("addq", [w, z]),
            Instr("movq", [y, tmp1]),
            Instr("negq", [tmp1]),
            Instr("movq", [z, tmp2]),
            Instr("addq", [tmp1, tmp2]),
            Instr("movq", [tmp2, Reg("rdi")]),
            Callq("print_int", 1),
        ]
        expected = [
            {v}, {v, w}, {w, x}, {w, x}, {w, x, y}, {w, y, z},
            {y, z}, {z, tmp1}, {z, tmp1}, {tmp1, tmp2}, {tmp2},
            {Reg("rdi")}, set(),
        ]
        self.assertEqual(Compiler().uncover_live(X86Program(instructions)), dict(enumerate(expected)))

    def test_two_operand_multiply_and_print_call(self):
        x, y = Variable("x"), Variable("y")
        rax, rdi = Reg("rax"), Reg("rdi")
        instructions = [
            Instr("movq", [Immediate(3), x]),
            Instr("movq", [Immediate(4), y]),
            Instr("movq", [x, rax]),
            Instr("imulq", [y, rax]),
            Instr("movq", [rax, rdi]),
            Callq("print_int", 1),
        ]
        expected = [
            {x}, {x, y}, {y, rax}, {rax}, {rdi}, set(),
        ]
        self.assertEqual(Compiler().uncover_live(X86Program(instructions)), dict(enumerate(expected)))

    def test_input_call_clobbers_return_register(self):
        x, y = Variable("x"), Variable("y")
        rax, rcx = Reg("rax"), Reg("rcx")
        instructions = [
            Instr("movq", [Immediate(1), Variable("unused")]),
            Callq("input_int", 0),
            Instr("movq", [rax, x]),
            Instr("movq", [x, rcx]),
            Instr("movq", [rcx, y]),
        ]
        self.assertEqual(
            Compiler().uncover_live(X86Program(instructions)),
            {0: set(), 1: {rax}, 2: {x}, 3: {rcx}, 4: set()},
        )

    def test_debug_prints_live_after_and_edges(self):
        v, w, x, y, z = map(Variable, ("v", "w", "x", "y", "z"))
        program = X86Program([
            Instr("movq", [Immediate(1), v]),
            Instr("movq", [Immediate(42), w]),
            Instr("movq", [v, x]),
            Instr("addq", [Immediate(7), x]),
            Instr("movq", [x, y]),
            Instr("movq", [x, z]),
            Instr("addq", [w, z]),
            Instr("addq", [y, z]),
            Instr("movq", [z, Reg("rdi")]),
            Callq("print_int", 1),
        ])
        output = StringIO()
        compiler = Compiler()
        with redirect_stderr(output):
            result = compiler.allocate_registers(program, debug=True)
        self.assertIs(result, program)
        lines = output.getvalue().splitlines()
        self.assertIn("Instruction", lines[0])
        self.assertIn("live-after", lines[0])
        self.assertIn("edges added", lines[0])
        self.assertTrue(lines[1].endswith("--"))
        self.assertTrue(lines[2].endswith("w-v"))
        self.assertTrue(lines[3].endswith("x-w"))
        self.assertTrue(lines[4].endswith("x-w"))
        self.assertTrue(lines[5].endswith("y-w"))
        self.assertTrue(lines[6].endswith("z-w, z-y"))
        self.assertTrue(lines[-1].endswith("--"))

    def test_interference_skips_move_source(self):
        v, w, x, y, z = map(Variable, ("v", "w", "x", "y", "z"))
        graph = self.graph_for([
            Instr("movq", [Immediate(1), v]),
            Instr("movq", [Immediate(42), w]),
            Instr("movq", [v, x]),
            Instr("movq", [x, y]),
            Instr("movq", [x, z]),
            Instr("addq", [w, z]),
            Instr("addq", [y, z]),
            Instr("movq", [z, Reg("rdi")]),
            Callq("print_int", 1),
        ])
        self.assertIn(w, graph[v])
        self.assertIn(w, graph[y])
        self.assertNotIn(x, graph[y])
        self.assertIn(y, graph[z])

    def test_call_interferes_with_caller_saved_registers(self):
        x = Variable("x")
        graph = self.graph_for([
            Instr("movq", [Immediate(1), x]),
            Callq("input_int", 0),
            Instr("movq", [x, Reg("rdi")]),
            Callq("print_int", 1),
        ])
        self.assertTrue({Reg("rax"), Reg("rdi"), Reg("r11")} <= graph[x])

    def test_saturation_coloring_respects_precolored_vertex(self):
        x, y, rax = Variable("x"), Variable("y"), Reg("rax")
        graph = {x: {y, rax}, y: {x, rax}, rax: {x, y}}
        colors = Compiler().color_graph(graph, (0, 1, 2), {rax: 0})
        self.assertEqual(colors[rax], 0)
        self.assertEqual({colors[x], colors[y]}, {1, 2})


if __name__ == "__main__":
    unittest.main()

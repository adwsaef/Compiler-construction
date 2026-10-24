from io import StringIO
import unittest

from compiler.compile import AVAILABLE_REGISTERS, Compiler
from x86_ast import Callq, Deref, Immediate, Instr, Reg, Variable, X86Program
from x86_interp import interp_x86


class AllocateRegistersTests(unittest.TestCase):
    def test_zero_register_cap_spills_and_executes(self):
        x, y = Variable("x"), Variable("y")
        program = X86Program([
            Instr("movq", [Immediate(2), x]),
            Instr("movq", [Immediate(3), y]),
            Instr("addq", [x, y]),
            Instr("movq", [y, Reg("rdi")]),
            Callq("print_int", 1),
        ])
        compiler = Compiler()
        allocated = compiler.allocate_registers(program, registers_cap=0)
        self.assertFalse(any(
            isinstance(operand, Variable)
            for instruction in allocated.body if isinstance(instruction, Instr)
            for operand in instruction.args
        ))
        self.assertTrue(any(
            isinstance(operand, Deref) and operand.offset <= -48
            for instruction in allocated.body if isinstance(instruction, Instr)
            for operand in instruction.args
        ))
        finished = compiler.prelude_and_conclusion(compiler.patch_instructions(allocated))
        self.assertEqual(interp_x86(finished, stdout=StringIO()), [5])

    def test_live_value_across_call_uses_callee_saved_register(self):
        x = Variable("x")
        program = X86Program([
            Instr("movq", [Immediate(7), x]),
            Callq("input_int", 0),
            Instr("addq", [Reg("rax"), x]),
            Instr("movq", [x, Reg("rdi")]),
            Callq("print_int", 1),
        ])
        compiler = Compiler()
        allocated = compiler.allocate_registers(program, registers_cap=1)
        self.assertTrue(any(
            Reg("rbx") in instruction.args
            for instruction in allocated.body if isinstance(instruction, Instr)
        ))
        finished = compiler.prelude_and_conclusion(compiler.patch_instructions(allocated))
        moves = [
            instruction.args for instruction in finished.body
            if isinstance(instruction, Instr) and instruction.instr == "movq"
        ]
        self.assertIn((Reg("rbx"), Deref("rbp", -8)), moves)
        self.assertIn((Deref("rbp", -8), Reg("rbx")), moves)
        self.assertEqual(
            interp_x86(finished, stdin=StringIO("5\n"), stdout=StringIO()),
            [12],
        )

    def test_register_cap_range(self):
        compiler = Compiler()
        for cap in (-1, len(AVAILABLE_REGISTERS) + 1):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                compiler.allocate_registers(X86Program([]), registers_cap=cap)


if __name__ == "__main__":
    unittest.main()

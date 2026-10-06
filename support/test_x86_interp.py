"""Tests for the x86 interpreter.

The end-to-end corpora in the stage directories are the interpreter's real
test suite; these cover what running a correct program cannot show -- the
diagnostics, the operand corner cases, and the arithmetic that differs
between Python and x86.
"""

import io

import pytest

from x86_ast import (
    ByteReg, Callq, Deref, Global, Immediate, Instr, Jump, JumpIf, Reg,
    Variable, X86Program,
)
from x86_interp import CODE_BASE, InterpError, Machine, interp_x86, load

# ---------------------------------------------------------------- helpers


def run(instrs, **kw):
    """Run a single 'main' block and return what it printed."""
    return interp_x86(X86Program({'main': list(instrs)}), **kw)


def run_blocks(blocks, **kw):
    return interp_x86(X86Program(dict(blocks)), **kw)


def mov(src, dst):
    return Instr('movq', [src, dst])


def imm(n):
    return Immediate(n)


PRINT = Callq('print_int', 1)


def prints(x):
    """movq x, %rdi ; callq print_int"""
    return [mov(x, Reg('rdi')), PRINT]


# ---------------------------------------------------------------- loading


def test_blocks_are_flattened_in_order():
    code, labels, where = load({'a': [mov(imm(1), Reg('rax'))],
                                'b': [mov(imm(2), Reg('rax'))]}, 'a')
    assert len(code) == 2
    assert labels == {'a': 0, 'b': 1}
    assert where[1] == ('b', 0)


def test_a_block_that_does_not_jump_falls_through():
    """Real assembly falls from one label into the next; so do we."""
    out = run_blocks({'main': [mov(imm(7), Reg('rdi'))],   # no jump
                      'next': [PRINT]})
    assert out == [7]


def test_undefined_jump_targets_are_reported_at_load():
    with pytest.raises(InterpError) as exc:
        load({'main': [Jump('nowhere'), JumpIf('e', 'elsewhere')]}, 'main')
    message = str(exc.value)
    # every undefined target, not just the first
    assert 'nowhere' in message and 'elsewhere' in message


def test_an_undefined_entry_label_is_reported():
    with pytest.raises(InterpError, match='start'):
        load({'main': []}, 'start')


# ---------------------------------------------------------------- operands


def test_an_unwritten_stack_slot_is_an_error_not_zero():
    with pytest.raises(InterpError, match=r'-8\(%rbp\)|uninitial'):
        run([mov(Deref('rbp', -8), Reg('rdi')), PRINT])


def test_an_unwritten_variable_is_an_error_naming_it():
    with pytest.raises(InterpError, match='x'):
        run([mov(Variable('x'), Reg('rdi')), PRINT])


def test_a_variable_round_trips():
    assert run([mov(imm(5), Variable('x'))] + prints(Variable('x'))) == [5]


def test_a_misaligned_offset_is_an_error():
    with pytest.raises(InterpError, match='align'):
        run([mov(imm(1), Deref('rbp', -4))])


def test_a_byte_register_aliases_the_low_byte_of_rax():
    out = run([mov(imm(0x1FF), Reg('rax')),
               mov(ByteReg('al'), Reg('rdi')), PRINT])
    assert out == [0xFF]


def test_writing_al_leaves_the_rest_of_rax_alone():
    out = run([mov(imm(0x1FF), Reg('rax')),
               mov(imm(0x02), ByteReg('al')),
               mov(Reg('rax'), Reg('rdi')), PRINT])
    assert out == [0x102]


def test_a_global_round_trips():
    out = run([mov(imm(99), Global('free_ptr'))] + prints(Global('free_ptr')))
    assert out == [99]


# ------------------------------------------------------------- arithmetic


def test_arithmetic_wraps_at_64_bits():
    out = run([mov(imm(2**63 - 1), Reg('rax')),
               Instr('addq', [imm(1), Reg('rax')])] + prints(Reg('rax')))
    assert out == [-2**63]


def test_multiplication_wraps_at_64_bits():
    out = run([mov(imm(2**62), Reg('rax')),
               Instr('imulq', [imm(4), Reg('rax')])] + prints(Reg('rax')))
    assert out == [0]


def test_division_truncates_toward_zero_as_x86_does():
    """Python's // floors: -7 // 2 == -4.  x86 gives -3."""
    out = run([mov(imm(-7), Reg('rax')),
               Instr('cqo', []),
               mov(imm(2), Reg('rcx')),
               Instr('idivq', [Reg('rcx')])] + prints(Reg('rax')))
    assert out == [-3]


def test_division_leaves_the_remainder_in_rdx():
    out = run([mov(imm(-7), Reg('rax')),
               Instr('cqo', []),
               mov(imm(2), Reg('rcx')),
               Instr('idivq', [Reg('rcx')])] + prints(Reg('rdx')))
    assert out == [-1]


def test_idivq_without_cqo_is_an_error():
    """%rdx must be the sign extension of %rax; a missing cqo is a bug."""
    with pytest.raises(InterpError, match='cqo|rdx'):
        run([mov(imm(-7), Reg('rax')),
             mov(imm(5), Reg('rdx')),
             mov(imm(2), Reg('rcx')),
             Instr('idivq', [Reg('rcx')])])


def test_negq_and_sarq():
    out = run([mov(imm(-40), Reg('rax')),
               Instr('negq', [Reg('rax')]),
               Instr('sarq', [imm(1), Reg('rax')])] + prints(Reg('rax')))
    assert out == [20]


def test_movzbq_zero_extends_one_byte_only():
    out = run([mov(imm(0x1FF), Reg('rax')),
               Instr('movzbq', [ByteReg('al'), Reg('rcx')])] + prints(Reg('rcx')))
    assert out == [0xFF]


# ------------------------------------------------------------------ flags


def test_movq_does_not_disturb_the_flags():
    """This is why leaq and movq are useful between a cmpq and a j<cc>."""
    m = Machine.for_test()
    m.flags = 'l'
    m.execute(mov(imm(1), Reg('rax')))
    assert m.flags == 'l'


def test_addq_does_set_the_flags():
    m = Machine.for_test()
    m.regs['rax'] = 5
    m.execute(Instr('addq', [imm(-5), Reg('rax')]))
    assert m.flags == 'e'


def test_cmpq_compares_the_second_operand_to_the_first():
    """cmpq b, a sets the flags for a <=> b -- the operand order is reversed."""
    m = Machine.for_test()
    m.regs['rax'] = 1
    m.execute(Instr('cmpq', [imm(2), Reg('rax')]))    # is 1 <=> 2
    assert m.flags == 'l'


def test_cmpq_is_right_when_the_difference_overflows():
    """a - b overflows 64 bits, but a < b is still false."""
    m = Machine.for_test()
    m.regs['rax'] = 2**63 - 1
    m.execute(Instr('cmpq', [imm(-1), Reg('rax')]))   # max <=> -1
    assert m.flags == 'g'


# ----------------------------------------------------------------- output


def test_print_int_writes_one_line_per_call():
    out = io.StringIO()
    result = run(prints(imm(1)) + prints(imm(2)), stdout=out)
    assert result == [1, 2]
    assert out.getvalue() == '1\n2\n'


def test_input_int_reads_a_line():
    out = run([Callq('input_int', 0), mov(Reg('rax'), Reg('rdi')), PRINT],
              stdin=io.StringIO('42\n'))
    assert out == [42]


# ------------------------------------------------------------ diagnostics


def test_an_unknown_mnemonic_says_so():
    with pytest.raises(InterpError, match='frobq'):
        run([Instr('frobq', [Reg('rax')])])


def test_a_32_bit_mnemonic_says_we_are_64_bit_only():
    with pytest.raises(InterpError, match='64-bit'):
        run([Instr('movl', [imm(1), Reg('rax')])])


def test_an_unsigned_condition_says_our_ints_are_signed():
    with pytest.raises(InterpError, match='signed'):
        run([Instr('cmpq', [imm(1), Reg('rax')]), Instr('seta', [ByteReg('al')])])


def test_the_step_budget_reports_where_it_stopped():
    with pytest.raises(InterpError, match='loop'):
        run_blocks({'main': [Jump('loop')], 'loop': [Jump('loop')]},
                   max_steps=100)


def test_the_entry_point_starts_with_the_abi_stack_alignment():
    """A function is entered with %rsp + 8 aligned to 16, because the
    caller's callq pushed a return address.  A prelude of just "pushq %rbp"
    is therefore correctly aligned for the call that follows it."""
    out = run_blocks({'main': [Instr('pushq', [Reg('rbp')]),
                               mov(Reg('rsp'), Reg('rbp')),
                               Callq('f', 0),
                               mov(imm(3), Reg('rdi')),
                               PRINT,
                               Instr('popq', [Reg('rbp')]),
                               Instr('retq', [])],
                      'f': [Instr('retq', [])]})
    assert out == [3]


def test_a_misaligned_frame_is_caught_at_the_call():
    with pytest.raises(InterpError, match='16-byte aligned'):
        run_blocks({'main': [Instr('pushq', [Reg('rbp')]),
                             Instr('pushq', [Reg('rbx')]),   # one push too many
                             Callq('f', 0)],
                    'f': [Instr('retq', [])]})


def test_a_retq_on_an_out_of_step_stack_says_so():
    """A pushq the callee never pops leaves retq reading data as an address."""
    with pytest.raises(InterpError, match='not a return address'):
        run_blocks({'main': [Instr('pushq', [Reg('rbp')]),
                             Callq('f', 0)],
                    'f': [Instr('pushq', [Reg('rbx')]),   # never popped
                          Instr('retq', [])]})


def test_falling_off_the_end_of_the_program_halts():
    assert run(prints(imm(1))) == [1]


# ------------------------------------------------- room for student tricks
#
# None of these is emitted by our compilers; all are things a student
# reasonably discovers.  See PLAN.md, "Room for students' ingenuity".


def test_leaq_adds_without_loading():
    out = run([mov(imm(40), Reg('rax')),
               Instr('leaq', [Deref('rax', 2), Reg('rcx')])] + prints(Reg('rcx')))
    assert out == [42]


def test_leaq_does_not_disturb_the_flags():
    """The other half of why leaq is worth reaching for."""
    m = Machine.for_test()
    m.regs['rax'] = 8
    m.execute(Instr('cmpq', [imm(9), Reg('rax')]))          # 8 <=> 9  ->  'l'
    m.execute(Instr('leaq', [Deref('rax', 8), Reg('rcx')]))
    assert m.flags == 'l' and m.regs['rcx'] == 16


def test_decq_and_jne_is_a_loop():
    """decq %rcx ; jne loop -- the idiom, and the reason flags generalise
    beyond cmpq."""
    out = run_blocks({
        'main': [mov(imm(5), Reg('rcx')), mov(imm(0), Reg('rax')),
                 Jump('loop')],
        'loop': [Instr('addq', [Reg('rcx'), Reg('rax')]),
                 Instr('decq', [Reg('rcx')]),
                 JumpIf('ne', 'loop'),
                 mov(Reg('rax'), Reg('rdi')), PRINT, Instr('retq', [])]})
    assert out == [15]                                       # 5+4+3+2+1


def test_a_shift_is_a_multiplication_by_a_power_of_two():
    out = run([mov(imm(21), Reg('rax')),
               Instr('salq', [imm(1), Reg('rax')])] + prints(Reg('rax')))
    assert out == [42]


def test_shrq_is_logical_and_sarq_is_arithmetic():
    out = run([mov(imm(-8), Reg('rbx')),
               Instr('sarq', [imm(1), Reg('rbx')]),
               mov(imm(-8), Reg('r12')),
               Instr('shrq', [imm(1), Reg('r12')])]
              + prints(Reg('rbx')) + prints(Reg('r12')))
    assert out == [-4, (2**64 - 8) >> 1]


def test_testq_is_the_idiomatic_zero_check():
    out = run_blocks({
        'main': [mov(imm(0), Reg('rax')),
                 Instr('testq', [Reg('rax'), Reg('rax')]),
                 JumpIf('e', 'zero'),
                 mov(imm(1), Reg('rdi')), PRINT, Instr('retq', [])],
        'zero': [mov(imm(0), Reg('rdi')), PRINT, Instr('retq', [])]})
    assert out == [0]


def test_cmov_replaces_a_branch():
    out = run([mov(imm(3), Reg('rax')), mov(imm(7), Reg('rcx')),
               Instr('cmpq', [Reg('rcx'), Reg('rax')]),      # 3 <=> 7
               Instr('cmovl', [Reg('rcx'), Reg('rax')])]     # rax = max
              + prints(Reg('rax')))
    assert out == [7]


def test_orq_notq_and_xchgq():
    out = run([mov(imm(0b1010), Reg('rbx')),
               Instr('orq', [imm(0b0101), Reg('rbx')]),
               mov(imm(0), Reg('r12')),
               Instr('notq', [Reg('r12')]),
               Instr('xchgq', [Reg('rbx'), Reg('r12')])]
              + prints(Reg('rbx')) + prints(Reg('r12')))
    assert out == [-1, 0b1111]


def test_scaled_index_addressing_says_it_is_our_limitation():
    from dataclasses import dataclass
    from x86_ast import arg

    @dataclass(frozen=True)
    class IndexedDeref(arg):        # what a student would have to invent
        base: str
        index: str
        scale: int

    with pytest.raises(InterpError, match='limitation of this interpreter'):
        run([Instr('leaq', [IndexedDeref('rax', 'rbx', 4), Reg('rcx')])])


# --------------------------------------------------------- pseudo-x86 mode


def test_a_program_without_a_prelude_says_which_entry_to_use():
    with pytest.raises(InterpError, match='main_start'):
        load({'main_start': [Instr('retq', [])]}, 'main')


def test_pseudo_x86_runs_with_variables_and_no_homes():
    """What gcc cannot do for us: run select_instructions' output."""
    out = run_blocks({'main_start': [mov(imm(40), Variable('a')),
                                     Instr('addq', [imm(2), Variable('a')]),
                                     mov(Variable('a'), Reg('rdi')),
                                     PRINT,
                                     Jump('main_conclusion')]},
                     entry='main_start')
    assert out == [42]


# ------------------------------------------- interpreting mid-pipeline
#
# Before prelude_and_conclusion has run there is no entry block, no frame and
# no callee-saved save/restore.  The interpreter stands in for the pass that
# has not run, so that this checkpoint tests the passes that have.


def test_a_call_before_the_prelude_finds_the_start_block():
    """Calls name f, but only f_start exists until the prelude is built."""
    out = run_blocks({'main_start': [mov(imm(1), Reg('rdi')),
                                     Callq('f', 1),
                                     Jump('main_conclusion')],
                      'f_start': [PRINT, Jump('f_conclusion')]})
    assert out == [1]


def test_a_jump_to_an_absent_conclusion_returns_rather_than_halting():
    out = run_blocks({'main_start': [Callq('f', 0), mov(imm(2), Reg('rdi')),
                                     PRINT, Jump('main_conclusion')],
                      'f_start': [mov(imm(1), Reg('rdi')), PRINT,
                                  Jump('f_conclusion')]})
    assert out == [1, 2]


def test_variables_are_local_to_a_frame():
    """A recursive callee must not overwrite its caller's variables."""
    out = run_blocks({
        'main_start': [mov(imm(2), Variable('n')), Callq('f', 0),
                       mov(Variable('n'), Reg('rdi')), PRINT,
                       Jump('main_conclusion')],
        'f_start': [mov(imm(99), Variable('n')), Jump('f_conclusion')]})
    assert out == [2]


def test_the_frame_size_comes_from_the_homes_the_code_uses():
    """Not an invented constant: the same number prelude_and_conclusion
    computes, from the deepest -N(%rbp) in the program."""
    from x86_interp import frame_size
    assert frame_size([mov(imm(0), Deref('rbp', -8))]) == 16
    assert frame_size([mov(imm(0), Deref('rbp', -40))]) == 48
    assert frame_size([mov(imm(0), Reg('rax'))]) == 0


def test_a_recursive_callee_does_not_overwrite_its_callers_spill_slots():
    """The frame is what the missing prelude would have allocated, so it goes
    on the stack -- if it did not, this would read f(0)'s slot, not f(2)'s."""
    out = run_blocks({
        'main_start': [mov(imm(2), Reg('rdi')), Callq('f', 1),
                       mov(Reg('rax'), Reg('rdi')), PRINT,
                       Jump('main_conclusion')],
        'f_start': [mov(Reg('rdi'), Deref('rbp', -8)),      # n, spilled
                    Instr('cmpq', [imm(0), Deref('rbp', -8)]),
                    JumpIf('e', 'f_base'),
                    mov(Deref('rbp', -8), Reg('rdi')),
                    Instr('subq', [imm(1), Reg('rdi')]),
                    Callq('f', 1),
                    # the recursive call must not have touched our slot
                    Instr('addq', [Deref('rbp', -8), Reg('rax')]),
                    Jump('f_conclusion')],
        'f_base': [mov(imm(0), Reg('rax')), Jump('f_conclusion')]})
    assert out == [3]                                        # 2 + 1 + 0


def test_the_synthetic_frame_lives_on_the_stack():
    """It is built by the prelude's own sequence, so %rsp moves and the saved
    %rbp is in memory, exactly as it would be in real assembly."""
    m = Machine.for_test()
    m.check_frames = False
    m.frame_size = 32
    rsp_before, rbp_before = m.regs['rsp'], m.regs['rbp']
    m.regs['rbx'] = 7
    m.build_frame()
    assert m.mem[m.regs['rbp']] == rbp_before        # saved %rbp is on the stack
    assert m.regs['rsp'] == m.regs['rbp'] - 8 * 5 - 32
    m.regs['rbx'] = 99
    m.tear_down_frame()
    assert (m.regs['rsp'], m.regs['rbp'], m.regs['rbx']) == (rsp_before,
                                                             rbp_before, 7)


def test_callee_saved_registers_survive_a_call_before_the_prelude():
    """The prelude would have pushed them; until it exists, we stand in."""
    out = run_blocks({
        'main_start': [mov(imm(7), Reg('rbx')), Callq('f', 0),
                       mov(Reg('rbx'), Reg('rdi')), PRINT,
                       Jump('main_conclusion')],
        'f_start': [mov(imm(99), Reg('rbx')), Jump('f_conclusion')]})
    assert out == [7]


def test_the_frame_check_waits_until_there_is_a_prelude():
    """%rsp alignment is what prelude_and_conclusion establishes, so there is
    nothing to check before it has run."""
    out = run_blocks({'main_start': [Callq('f', 0), Jump('main_conclusion')],
                      'f_start': [mov(imm(1), Reg('rdi')), PRINT,
                                  Jump('f_conclusion')]})
    assert out == [1]


def test_the_entry_is_found_without_being_named():
    assert run_blocks({'main_start': [mov(imm(4), Reg('rdi')), PRINT,
                                      Jump('main_conclusion')]}) == [4]


# ============================================================ review fixes


def test_a_value_left_in_a_caller_saved_register_across_a_call_is_caught():
    """Lecture 3's whole subject: %rcx does not survive a call."""
    with pytest.raises(InterpError, match='caller-saved|destroyed'):
        run(prints(imm(222)) + [mov(imm(111), Reg('rcx'))] + prints(imm(222))
            + [mov(Reg('rcx'), Reg('rdi')), PRINT])


def test_a_callee_saved_register_does_survive_a_call():
    out = run([mov(imm(7), Reg('rbx'))] + prints(imm(1))
              + [mov(Reg('rbx'), Reg('rdi')), PRINT])
    assert out == [1, 7]


def test_the_result_register_is_readable_after_a_call():
    out = run([Callq('input_int', 0), mov(Reg('rax'), Reg('rdi')), PRINT],
              stdin=io.StringIO('5\n'))
    assert out == [5]


def test_a_store_through_an_uninitialised_pointer_is_rejected():
    """%r11 is still 0, so this writes to 0x8 -- in no region we know."""
    with pytest.raises(InterpError, match='no region|not in'):
        run([mov(imm(42), Deref('r11', 8))])


def test_a_store_past_the_end_of_the_heap_is_rejected():
    with pytest.raises(InterpError, match='heap|no region'):
        run([mov(Global('fromspace_end'), Reg('r11')),
             mov(imm(1), Deref('r11', 8))])


def test_an_unmodelled_byte_register_is_rejected_not_aliased():
    """setl %sil must not quietly read %al and write %rax."""
    with pytest.raises(InterpError, match='sil'):
        run([mov(imm(1), ByteReg('sil'))])


def test_an_indirect_call_returns_to_its_caller():
    """It is a call, not a jump: without a return address the callee's retq
    goes to the caller's caller and the rest of main is silently skipped."""
    from x86_ast import IndirectCallq
    from x86_interp import Machine, load

    blocks = {'main': [Instr('pushq', [Reg('rbp')]),   # a prelude, so that
                       IndirectCallq(Reg('rax'), 0),   # %rsp is aligned
                       mov(imm(99), Reg('rdi')), PRINT,
                       Instr('popq', [Reg('rbp')]), Instr('retq', [])],
              'f': [mov(imm(7), Reg('rdi')), PRINT, Instr('retq', [])]}
    code, labels, where = load(blocks, 'main')
    m = Machine(code, labels, where, 'main', stdout=io.StringIO())
    m.regs['rax'] = CODE_BASE + 8 * labels['f']       # a function's address
    assert m.run(1000) == [7, 99]


def test_three_operand_imulq():
    out = run([mov(imm(6), Reg('rcx')),
               Instr('imulq', [imm(7), Reg('rcx'), Reg('rax')])]
              + prints(Reg('rax')))
    assert out == [42]


def test_a_call_to_an_undefined_function_says_so():
    with pytest.raises(InterpError, match='nosuchfn'):
        run_blocks({'main': [Callq('nosuchfn', 0)]})


def test_a_shift_count_is_masked_to_six_bits_as_x86_does():
    """salq $64 is a no-op on x86, not a zeroing."""
    out = run([mov(imm(3), Reg('rax')),
               Instr('salq', [imm(64), Reg('rax')])] + prints(Reg('rax')))
    assert out == [3]


def test_a_negative_shift_count_is_a_sited_error():
    with pytest.raises(InterpError, match='shift'):
        run([mov(imm(3), Reg('rax')), Instr('salq', [imm(-1), Reg('rax')])])


def test_addq_sets_the_flags_from_the_unwrapped_result_like_cmpq():
    """Overflow does not make a positive sum look negative."""
    m = Machine.for_test()
    m.regs['rax'] = 2**63 - 1
    m.execute(Instr('addq', [imm(1), Reg('rax')]))
    assert m.regs['rax'] == -2**63        # the value wraps ...
    assert m.flags == 'g'                 # ... the flags do not lie about it


def test_non_numeric_input_is_a_sited_error():
    with pytest.raises(InterpError, match='input_int'):
        run([Callq('input_int', 0)], stdin=io.StringIO('hello\n'))


# ================================== the caller-saved check does not overreach


def test_setcc_and_movzbq_right_after_a_call_are_accepted():
    """What every "print(x); print(1 if x < 5 else 0)" compiles to.

    print_int destroys %rax, but set<cc> defines the low byte and movzbq
    reads only that byte, so the destroyed upper bytes are never observed.
    """
    out = run([mov(imm(3), Reg('rbx'))] + prints(Reg('rbx'))
              + [Instr('cmpq', [imm(5), Reg('rbx')]),
                 Instr('setl', [ByteReg('al')]),
                 Instr('movzbq', [ByteReg('al'), Reg('rcx')]),
                 mov(Reg('rcx'), Reg('rdi')), PRINT])
    assert out == [3, 1]


def test_the_remainder_is_readable_after_a_call_and_a_cqo():
    """cqo defines %rdx and idivq redefines it, so neither is stale."""
    out = (run(prints(imm(1))
               + [mov(imm(17), Reg('rax')), mov(imm(5), Reg('rcx')),
                  Instr('cqo', []), Instr('idivq', [Reg('rcx')]),
                  mov(Reg('rdx'), Reg('rdi')), PRINT]))
    assert out == [1, 2]


def test_a_caller_saved_register_as_an_address_base_is_caught():
    """"movq 0(%rcx), %rax" is "movq %rcx, %rax" with a memory operand."""
    with pytest.raises(InterpError, match='caller-saved|destroyed'):
        run([mov(Reg('rsp'), Reg('rcx'))] + prints(imm(1))
            + [mov(Deref('rcx', 0), Reg('rax'))])


def test_an_error_inside_a_binary_instruction_is_sited_once():
    with pytest.raises(InterpError) as exc:
        run([Instr('salq', [imm(-1), Reg('rax')])])
    assert str(exc.value).count('  at ') == 1

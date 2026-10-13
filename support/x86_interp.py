"""An interpreter for the x86-64 subset this course generates.

It runs an X86Program -- the same one the compilers build -- rather than
assembly text, which means it can also run the output of select_instructions,
while the operands are still Variables and before any homes exist.  That is
the thing gcc cannot do for us.

Not a general-purpose emulator.  It executes what is needed for the course,
plus some instructions that can be used in peephole optimisation (leaq, incq/decq,
 shifts, testq, cmov), and refuses everything else with a message saying why.

Important simplifications:
  - the program counter is an index into one flat instruction array,
    not a byte counter; jumps to middle of instructions are not supported
  - memory is word-addressed and sparse, and reading a word that was never
    written is an error rather than a zero or garbage value;
  - flags are a single three-way value, the sign of the last ALU result.
"""

import sys
from typing import Callable, TextIO

from x86_ast import (
    ByteReg, Callq, Deref, Global, Immediate, IndirectCallq, IndirectJump,
    Instr, Jump, JumpIf, Reg, Variable, X86Def, X86Program, arg, instr,
)

# Three disjoint regions, so that a value says what it is.
CODE_BASE = 0x0040_0000       # only executed as code, never read as data
STACK_TOP = 0x7fff_0000       # the stack grows down from here
HEAP_BASE = 0x1000_0000       # ... and the heap lives well below it
ROOTSTACK_BASE = 0x0800_0000  # for the allocator and garbage collector

WORD = 8
MASK64 = (1 << 64) - 1

# The registers a prelude is responsible for preserving.
CALLEE_SAVED = ['rbx', 'r12', 'r13', 'r14', 'r15']

# The ones a call may destroy.  %rax is not in here because it carries the
# result back; a compiler that reads it after a void call has a bug we cannot
# see without knowing the callee's return type.
CALLER_SAVED = ['rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11']

STACK_SIZE = 8 << 20
ROOTSTACK_SIZE = 1 << 16

REGISTERS = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
             'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']

# The only byte registers we model.  An unmodelled one must raise rather than
# quietly aliasing %rax, which would corrupt %rax and leave the intended
# register untouched, with no complaint.
BYTE_PARENT = {'al': 'rax', 'bl': 'rbx', 'cl': 'rcx', 'dl': 'rdx'}

# Condition codes, and the flag values that satisfy them.  One table for
# j<cc>, set<cc> and cmov<cc> alike.
SATISFIED = {'e': {'e'}, 'ne': {'l', 'g'},
             'l': {'l'}, 'le': {'l', 'e'},
             'g': {'g'}, 'ge': {'g', 'e'}}

# Recognised, but deliberately not supported: our only numeric type is a
# signed 64-bit integer, so an unsigned branch is a bug, not a shortcut.
UNSIGNED_CC = {'a', 'ae', 'b', 'be', 'na', 'nae', 'nb', 'nbe', 'c', 'nc'}

# Everything execute() handles, for the "did you mean the 64-bit form?" hint.
MNEMONICS = {'movq', 'cmpq', 'leaq', 'cqo', 'cqto', 'idivq', 'movzbq',
             'xchgq', 'pushq', 'popq', 'retq', 'addq', 'subq', 'imulq',
             'andq', 'orq', 'xorq', 'sarq', 'salq', 'shlq', 'shrq', 'testq',
             'negq', 'notq', 'incq', 'decq'}

RUNTIME = {'print_int', 'input_int', 'read_int'}
NOT_MODELLED = {
    'collect': 'this interpreter does not model garbage collection yet.\n'
               'Give it a larger heap (heap_size=) so that collection is '
               'not reached.',
    'initialize': 'this interpreter sets up the heap itself; there is '
                  'nothing for initialize to do.',
}


class InterpError(Exception):
    """Anything the interpreter refuses to do, with a reason."""


def to_signed(x: int) -> int:
    return ((x + (1 << 63)) & MASK64) - (1 << 63)


def sign(x: int) -> str:
    return 'e' if x == 0 else ('l' if x < 0 else 'g')


def shift_count(amount: int) -> int:
    """x86 masks a 64-bit shift count to 6 bits, so salq $64 is a no-op."""
    if amount < 0:
        raise InterpError(f'negative shift count {amount}')
    return amount & 0x3F


def shift_right_logical(value: int, amount: int) -> int:
    """shrq: on the 64-bit pattern, not on a Python integer."""
    return (value & MASK64) >> shift_count(amount)


# name -> (compute(dst, src), writes the destination, sets the flags)
#
# Every entry reads *both* operands.  movq does not read its destination, so
# it is not in here -- putting it here would make "movq $5, -8(%rbp)" report
# the slot as read before written.  Nor is cmpq, which has no destination and
# must compare without wrapping.
BINARY: dict[str, tuple[Callable[[int, int], int], bool, bool]] = {
    'addq':  (lambda d, s: d + s,   True,  True),
    'subq':  (lambda d, s: d - s,   True,  True),
    'imulq': (lambda d, s: d * s,   True,  True),
    'andq':  (lambda d, s: d & s,   True,  True),
    'orq':   (lambda d, s: d | s,   True,  True),
    'xorq':  (lambda d, s: d ^ s,   True,  True),
    'sarq':  (lambda d, s: d >> shift_count(s),  True, True),
    'salq':  (lambda d, s: d << shift_count(s),  True, True),
    'shlq':  (lambda d, s: d << shift_count(s),  True, True),
    'shrq':  (shift_right_logical,  True,  True),
    'testq': (lambda d, s: d & s,   False, True),
    # cmpq is the odd one out: no destination, and it must compare without
    # wrapping.  It is handled in execute().
}

# name -> (compute(value), sets the flags)
UNARY: dict[str, tuple[Callable[[int], int], bool]] = {
    'negq': (lambda v: -v,    True),
    'notq': (lambda v: ~v,    False),   # notq is the one that does not
    'incq': (lambda v: v + 1, True),
    'decq': (lambda v: v - 1, True),
}


# ---------------------------------------------------------------- loading


Program = X86Program | X86Def | list


def blocks_of(prog: Program) -> dict[str, list[instr]]:
    """The block dictionary of a program, however it is shaped.

    The shapes come from different points in the pipeline: an X86Program with
    a flat body before there are blocks, one with a block dict after
    explicate_control, an X86Def or a list of them while the back end is
    running one function at a time.
    """
    if isinstance(prog, list):
        blocks: dict[str, list[instr]] = {}
        for definition in prog:
            blocks.update(blocks_of(definition))
        return blocks
    if isinstance(prog, X86Def):
        return dict(prog.blocks)
    if isinstance(prog.body, dict):
        return dict(prog.body)
    return {'main': list(prog.body)}


def frame_size(code: list[instr]) -> int:
    """How much frame the deepest -N(%rbp) home in this program needs.

    prelude_and_conclusion works this out per function and emits the subq for
    it.  When we are asked to interpret a program from before that pass, we
    have to stand in for it -- so derive the number the same way, from the
    homes the code actually uses, rather than inventing one.  Taking the
    maximum over the whole program rather than per function is a safe
    over-estimate; the blocks do not say which function they belong to.
    """
    deepest = 0
    for ins in code:
        if isinstance(ins, Instr):
            for a in ins.args:
                if isinstance(a, Deref) and a.reg == 'rbp' and a.offset < 0:
                    deepest = max(deepest, -a.offset)
    return (deepest + 15) // 16 * 16


def default_entry(blocks: dict[str, list[instr]]) -> str:
    """Where to start, when the caller has not said.

    Before prelude_and_conclusion there is no `main` block -- the program's
    own statements are in `main_start` -- so which one exists tells us how far
    down the pipeline this program is.
    """
    for candidate in ('main', 'main_start'):
        if candidate in blocks:
            return candidate
    raise InterpError(
        'no entry block: expected main or main_start, found '
        + ', '.join(sorted(blocks)))


def load(blocks: dict[str, list[instr]], entry: str = 'main'
         ) -> tuple[list[instr], dict[str, int], dict[int, tuple[str, int]]]:
    """Flatten the blocks into one array and resolve the labels.

    Returns (code, labels, where); `where` maps an index back to
    (label, offset), for diagnostics only.  Every undefined jump target is
    reported here, before execution starts, rather than one at a time as
    execution happens to reach them.
    """
    code: list[instr] = []
    labels: dict[str, int] = {}
    where: dict[int, tuple[str, int]] = {}

    for label, body in blocks.items():
        labels[label] = len(code)
        for offset, ins in enumerate(body):
            where[len(code)] = (label, offset)
            code.append(ins)

    undefined = [entry] if entry not in labels else []
    for ins in code:
        match ins:
            case Jump(label) | JumpIf(_, label):
                # A conclusion block is often left implicit; jumping to one
                # is how a function returns.
                if label in labels or label.endswith('conclusion'):
                    continue
                undefined.append(label)
            case Callq(func, _) if func not in RUNTIME \
                    and func not in NOT_MODELLED:
                # Before prelude_and_conclusion a function has no entry
                # block: calls name `f` while only `f_start` exists.  But a
                # call to something that is neither is an error here rather
                # than a jump to nowhere at run time.
                if func in labels or func + '_start' in labels:
                    continue
                undefined.append(func)

    if undefined:
        message = ('undefined label(s): ' + ', '.join(sorted(set(undefined)))
                   + '\n  known labels: ' + ', '.join(sorted(labels)))
        if entry in undefined and entry + '_start' in labels:
            # The prelude has not been built yet, so there is no entry block.
            message += (f'\n  {entry}_start exists: this program has not '
                        f'reached prelude_and_conclusion, so pass '
                        f'entry="{entry}_start".')
        raise InterpError(message)

    return code, labels, where


# ------------------------------------------------------------- the machine


class Machine:
    def __init__(self, code: list[instr], labels: dict[str, int],
                 where: dict[int, tuple[str, int]], entry: str = 'main', *,
                 stdin: TextIO | None = None, stdout: TextIO | None = None,
                 heap_size: int = 1 << 20, check_frames: bool = True):
        self.code = code
        self.labels = labels
        self.where = where
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout

        self.regs: dict[str, int] = {r: 0 for r in REGISTERS}
        # The ABI hands a function %rsp + 8 aligned to 16, because the
        # caller's callq has just pushed a return address.  Model that push
        # for the entry point, or every prelude computed from it is 8 out.
        self.regs['rsp'] = STACK_TOP - WORD
        self.regs['rbp'] = STACK_TOP - WORD
        self.mem: dict[int, int] = {}
        # Variables are function-local, so a call gets a fresh set: arguments
        # arrive in registers, and the result leaves in %rax.  Without this a
        # recursive call would overwrite its caller's variables -- which only
        # matters before homes are assigned, but that is exactly when this
        # interpreter is most useful.
        self.vars: dict[str, int] = {}
        self.var_frames: list[dict[str, int]] = []
        self.frame_size = frame_size(code)
        self.synthetic_frames = 0
        self.globals: dict[str, int] = {
            'free_ptr': HEAP_BASE,
            'fromspace_begin': HEAP_BASE,
            'fromspace_end': HEAP_BASE + heap_size,
            # Seeded for the collector we do not model yet (PLAN.md D10).
            'rootstack_begin': ROOTSTACK_BASE,
        }
        self.flags: str | None = None
        # Registers a call has destroyed.  Reading one is an error: keeping a
        # value in a caller-saved register across a call is lecture 3's
        # subject, and it is the class of bug an interpreter that models the
        # runtime as transparent cannot see.
        self.clobbered: set[str] = set()
        self.output: list[int] = []
        self.pc = labels.get(entry, 0)
        self.halted = False
        # The 16-byte frame invariant is what prelude_and_conclusion
        # establishes, so there is nothing to check before that pass has run.
        self.check_frames = check_frames

    @classmethod
    def for_test(cls) -> 'Machine':
        """A machine with no code, for exercising single instructions."""
        return cls([], {}, {})

    # -- diagnostics ------------------------------------------------------

    def site(self) -> str:
        if self.pc in self.where:
            label, offset = self.where[self.pc]
            return f'{label}+{offset}'
        return f'pc={self.pc}'

    def fail(self, message: str) -> InterpError:
        return InterpError(f'{message}\n  at {self.site()}')

    def byte_parent(self, rid: str) -> str:
        if rid not in BYTE_PARENT:
            raise self.fail(
                f'%{rid} is not a byte register this interpreter models '
                '(' + ', '.join('%' + r for r in BYTE_PARENT) + ')')
        return BYTE_PARENT[rid]

    def region_of(self, addr: int) -> str | None:
        """Which of the modelled regions an address is in, if any."""
        if STACK_TOP - STACK_SIZE <= addr <= STACK_TOP:
            return 'stack'
        if HEAP_BASE <= addr < self.globals['fromspace_end']:
            return 'heap'
        if ROOTSTACK_BASE <= addr < ROOTSTACK_BASE + ROOTSTACK_SIZE:
            return 'root stack'
        if CODE_BASE <= addr < CODE_BASE + WORD * (len(self.code) + 1):
            return 'code'
        return None

    def check_region(self, a: arg, addr: int) -> None:
        """Refuse an access that lands outside every region we model.

        Registers start at 0, so without this a store through a pointer that
        was never loaded lands at a small address, succeeds, and reads back --
        the value round-trips through null and the program may even print the
        right answer.  That is the mirror of the read-before-written rule, and
        the characteristic failure of the tuples stage.
        """
        region = self.region_of(addr)
        if region is None:
            raise self.fail(
                f'{a} (address {addr:#x}) is in no region this interpreter '
                'models.\n'
                f'  stack {STACK_TOP - STACK_SIZE:#x}..{STACK_TOP:#x}, '
                f'heap {HEAP_BASE:#x}..'
                f'{self.globals["fromspace_end"]:#x}, '
                f'root stack {ROOTSTACK_BASE:#x}..'
                f'{ROOTSTACK_BASE + ROOTSTACK_SIZE:#x}.\n'
                '  A pointer that was never loaded reads as 0, which lands '
                'here.')
        if region == 'code':
            raise self.fail(
                f'{a} (address {addr:#x}) is a return address, not data')

    def region_hint(self, addr: int) -> str:
        region = self.region_of(addr)
        return f' -- that address is in the {region}' if region else ''

    @staticmethod
    def unknown_operand(a: arg) -> str:
        name = type(a).__name__
        if 'Index' in name or 'Scaled' in name:
            return ('scaled-index addressing is a limitation of this '
                    'interpreter, not of x86: our operands are '
                    'base-plus-offset only.\n'
                    'Use an explicit imulq or salq and an addq.')
        return (f'{name} is not an operand this interpreter models '
                '(immediate, register, %al, offset(%reg), global, variable)')

    def unknown_mnemonic(self, name: str) -> str:
        if name.startswith(('set', 'j', 'cmov')):
            cc = name.removeprefix('set').removeprefix('cmov').removeprefix('j')
            if cc in UNSIGNED_CC:
                return (f'{name}: unsigned comparison -- every value in our '
                        'language is a signed 64-bit integer, so this is a '
                        'bug rather than a shortcut.')
        if name.endswith(('l', 'w', 'b')) and name[:-1] + 'q' in MNEMONICS:
            return (f'{name}: this interpreter models 64-bit operations only '
                    f'(every value in our language is a 64-bit signed '
                    f'integer).  Did you mean {name[:-1]}q?')
        return f'{name}: not an instruction this interpreter models'

    # -- operands ---------------------------------------------------------

    def address(self, a: arg, *, for_access: bool = True) -> int:
        """The address a memory operand denotes, without reading it.

        `for_access` is False for leaq, which computes an address rather than
        using one -- "leaq 2(%rax), %rcx" is a perfectly good way to add 2,
        and need not land on a word boundary.
        """
        match a:
            case Deref(reg, offset):
                if reg not in self.regs:
                    raise self.fail(f'no such register: %{reg}')
                # Through read(), so that a base register the call above may
                # have destroyed is caught here too: "movq 0(%rcx), %rax" is
                # the same mistake as "movq %rcx, %rax", spelled differently.
                addr = to_signed(self.read(Reg(reg)) + offset)
                if for_access:
                    if addr % WORD:
                        raise self.fail(
                            f'{a} is not 8-byte aligned (address {addr:#x}); '
                            'every value we store is a 64-bit word')
                    self.check_region(a, addr)
                return addr
            case Immediate() | Reg() | Variable() | Global():
                raise self.fail(f'{a} is not a memory operand')
            case _:
                raise self.fail(self.unknown_operand(a))

    def read(self, a: arg) -> int:
        match a:
            case Immediate(value):
                return to_signed(value)
            case ByteReg(rid):
                # Deliberately not through read(Reg(...)): the byte path does
                # not consult the caller-saved bookkeeping.  After a call,
                # "setl %al; movzbq %al, dst" is correct -- setl defines the
                # low byte and movzbq reads only that byte, so the destroyed
                # upper 56 bits are never observed.
                return self.regs[self.byte_parent(rid)] & 0xFF
            case Reg(rid):
                if rid not in self.regs:
                    raise self.fail(f'no such register: %{rid}')
                if rid in self.clobbered:
                    raise self.fail(
                        f'%{rid} is read here, but the call above may have '
                        f'destroyed it: %{rid} is caller-saved.\n'
                        '  A value that has to survive a call belongs in a '
                        'callee-saved register (' + ', '.join(
                            '%' + r for r in CALLEE_SAVED) + ') or on the '
                        'stack.')
                return self.regs[rid]
            case Variable(name):
                if name not in self.vars:
                    raise self.fail(
                        f'variable {name} is read before it is written')
                return self.vars[name]
            case Deref():
                addr = self.address(a)
                if addr not in self.mem:
                    raise self.fail(
                        f'{a} (address {addr:#x}) is read before it is '
                        f'written{self.region_hint(addr)}')
                return self.mem[addr]
            case Global(name):
                if name not in self.globals:
                    raise self.fail(f'no such global: {name}')
                return self.globals[name]
            case _:
                raise self.fail(self.unknown_operand(a))

    def write(self, a: arg, value: int) -> None:
        value = to_signed(value)
        match a:
            case ByteReg(rid):
                # Writing the low byte defines what the following movzbq
                # reads, so it clears the caller-saved mark rather than
                # tripping it.  Strictly the upper bytes are still garbage;
                # we do not track definedness per byte, because
                # "set<cc> %al; movzbq %al, dst" is the only shape we emit.
                parent = self.byte_parent(rid)
                self.clobbered.discard(parent)
                self.regs[parent] = to_signed(
                    (self.regs[parent] & ~0xFF) | (value & 0xFF))
            case Reg(rid):
                if rid not in self.regs:
                    raise self.fail(f'no such register: %{rid}')
                self.clobbered.discard(rid)
                self.regs[rid] = value
            case Variable(name):
                self.vars[name] = value
            case Deref():
                self.mem[self.address(a)] = value
            case Global(name):
                self.globals[name] = value
            case Immediate():
                raise self.fail('cannot write to an immediate')
            case _:
                raise self.fail(self.unknown_operand(a))

    # -- control ----------------------------------------------------------

    def condition(self, cc: str) -> bool:
        if cc in UNSIGNED_CC:
            raise self.fail(
                f'condition code "{cc}": unsigned comparison -- every value '
                'in our language is a signed 64-bit integer, so this is a '
                'bug rather than a shortcut.')
        if cc not in SATISFIED:
            raise self.fail(f'unknown condition code: {cc}')
        if self.flags is None:
            raise self.fail(
                'the flags are read here, but nothing has set them -- '
                'a comparison is missing before this')
        return self.flags in SATISFIED[cc]

    def target(self, label: str) -> int | None:
        """The pc a jump to `label` lands on; None means carry on.

        A conclusion block that does not exist means prelude_and_conclusion
        has not run yet.  Jumping to one is how a function returns, so do
        what the epilogue would: resume in the caller, or halt if this is the
        outermost frame.
        """
        if label in self.labels:
            return self.labels[label]
        return self.do_return()

    def clobber_caller_saved(self, also_rax: bool = False) -> None:
        """A call may destroy these; reading one afterwards is an error."""
        self.clobbered.update(CALLER_SAVED)
        if also_rax:
            self.clobbered.add('rax')

    def call(self, func: str) -> int | None:
        if func in NOT_MODELLED:
            raise self.fail(f'{func}: {NOT_MODELLED[func]}')
        if func == 'print_int':
            value = self.read(Reg('rdi'))
            self.output.append(value)
            print(value, file=self.stdout)      # runtime.c: printf("%ld\n")
            self.clobber_caller_saved(also_rax=True)   # print_int is void
            return None
        if func in ('input_int', 'read_int'):
            line = self.stdin.readline()
            if not line:
                raise self.fail(f'{func}: no more input')
            try:
                value = int(line.strip())
            except ValueError:
                raise self.fail(
                    f'{func}: {line.strip()!r} is not an integer') from None
            self.clobber_caller_saved(also_rax=True)
            self.write(Reg('rax'), value)              # ... and then returns it
            return None
        target = self.labels.get(func, self.labels.get(func + '_start'))
        if target is None:
            raise self.fail(
                f'callq {func}: no such function.\n'
                '  known labels: ' + ', '.join(sorted(self.labels)))
        return self.call_to(target, func)

    def build_frame(self) -> None:
        """Do what the missing prelude would have done, on the real stack.

        Before prelude_and_conclusion, nothing allocates a frame or preserves
        the callee-saved registers, so a nested call would overwrite its
        caller's spill slots and registers.  Rather than book-keeping that
        beside the machine, run the prelude's own instruction sequence -- the
        frame then lives on the stack, where assembly puts it, and a memory
        dump shows what a real frame would show.
        """
        self.push(self.regs['rbp'])
        self.regs['rbp'] = self.regs['rsp']
        for name in CALLEE_SAVED:
            self.push(self.regs[name])
        self.regs['rsp'] = to_signed(self.regs['rsp'] - self.frame_size)
        self.synthetic_frames += 1

    def tear_down_frame(self) -> None:
        """The conclusion's half: undo build_frame, in reverse."""
        self.regs['rsp'] = to_signed(
            self.regs['rbp'] - WORD * len(CALLEE_SAVED))
        for name in reversed(CALLEE_SAVED):
            self.regs[name] = self.pop()
        self.regs['rbp'] = self.pop()
        self.synthetic_frames -= 1

    def push(self, value: int) -> None:
        self.regs['rsp'] = to_signed(self.regs['rsp'] - WORD)
        self.mem[self.regs['rsp']] = value

    def pop(self) -> int:
        addr = self.regs['rsp']
        if addr not in self.mem:
            raise self.fail('popq from an address never pushed to')
        self.regs['rsp'] = to_signed(addr + WORD)
        return self.mem[addr]

    def call_to(self, target: int, func: str = '*') -> int:
        """The bookkeeping a callq does, given a resolved target."""
        if self.check_frames and self.regs['rsp'] % 16:
            raise self.fail(
                f'callq {func} with %rsp = {self.regs["rsp"]:#x}, which is '
                'not 16-byte aligned.\n'
                '  The ABI requires %rsp to be a multiple of 16 immediately '
                'before a callq, so that the callee sees %rsp + 8 aligned.\n'
                '  Count what the prelude pushed: %rbp, the callee-saved '
                'registers, and the subq for the spill area.')
        self.push(CODE_BASE + WORD * (self.pc + 1))
        self.var_frames.append(self.vars)
        self.vars = {}
        if not self.check_frames:
            self.build_frame()
        return target

    def do_return(self) -> int | None:
        if self.synthetic_frames:
            self.tear_down_frame()
        addr = self.regs['rsp']
        if addr not in self.mem:
            self.halted = True      # retq from the outermost frame
            return None
        value = self.mem[addr]
        self.regs['rsp'] = to_signed(addr + WORD)
        target = self.pc_of_address(value)
        if self.var_frames:
            self.vars = self.var_frames.pop()
        # Back in the caller: the callee-saved registers have been restored,
        # %rax carries the result, and everything else is gone.
        self.clobbered = set(CALLER_SAVED)
        return target

    def pc_of_address(self, value: int) -> int:
        offset = value - CODE_BASE
        if offset < 0 or offset % WORD or offset // WORD > len(self.code):
            raise self.fail(
                f'{value:#x} is not a return address; the stack is out of '
                'step -- a pushq without its popq, or the wrong frame size')
        return offset // WORD

    # -- one instruction --------------------------------------------------

    def execute(self, ins: instr) -> int | None:
        """Perform one instruction; return the new pc, or None for pc + 1."""
        match ins:
            case Instr('movq', [src, dst]):
                # Does not read its destination, and does not set the flags.
                self.write(dst, self.read(src))
                return None

            case Instr('cmpq', [src, dst]):
                # cmpq b, a sets the flags for a <=> b -- the operand order
                # is reversed.  The difference is taken without wrapping, so
                # jl is right even when a - b overflows; real hardware gets
                # the same answer from SF != OF.
                self.flags = sign(self.read(dst) - self.read(src))
                return None

            case Instr('leaq', [src, dst]):
                # Computes the address instead of loading from it -- and, the
                # other half of why it is useful, leaves the flags alone.
                self.write(dst, self.address(src, for_access=False))
                return None

            case Instr('cqo' | 'cqto', []):
                self.write(Reg('rdx'), -1 if self.read(Reg('rax')) < 0 else 0)
                return None

            case Instr('idivq', [src]):
                self.divide(self.read(src))
                return None

            case Instr('imulq', [src, mul, dst]):
                # The three-operand form: dst = src * mul, leaving mul alone.
                result = to_signed(self.read(src) * self.read(mul))
                self.write(dst, result)
                self.flags = sign(result)
                return None

            case Instr('movzbq', [src, dst]):
                self.write(dst, self.read(src) & 0xFF)
                return None

            case Instr('xchgq', [a, b]):
                va, vb = self.read(a), self.read(b)
                self.write(a, vb)
                self.write(b, va)
                return None

            case Instr('pushq', [src]):
                value = self.read(src)
                self.regs['rsp'] = to_signed(self.regs['rsp'] - WORD)
                self.mem[self.regs['rsp']] = value
                return None

            case Instr('popq', [dst]):
                addr = self.regs['rsp']
                if addr not in self.mem:
                    raise self.fail('popq from an address never pushed to')
                self.write(dst, self.mem[addr])
                self.regs['rsp'] = to_signed(addr + WORD)
                return None

            case Instr('retq', []):
                return self.do_return()

            case Instr(name, [src, dst]) if name in BINARY:
                compute, writes, sets_flags = BINARY[name]
                # The reads site their own errors; only compute's need it.
                lhs, rhs = self.read(dst), self.read(src)
                try:
                    exact = compute(lhs, rhs)
                except InterpError as exc:
                    raise self.fail(str(exc)) from None
                if writes:
                    self.write(dst, to_signed(exact))
                if sets_flags:
                    # From the unwrapped result, as cmpq does: overflow makes
                    # the stored value wrap, but it must not make the flags
                    # lie about the sign.  Real hardware reads SF != OF here.
                    self.flags = sign(exact)
                return None

            case Instr(name, [dst]) if name in UNARY:
                compute_one, one_sets_flags = UNARY[name]
                exact = compute_one(self.read(dst))
                self.write(dst, to_signed(exact))
                if one_sets_flags:
                    self.flags = sign(exact)
                return None

            case Instr(name, [dst]) if name.startswith('set'):
                self.write(dst, 1 if self.condition(name[3:]) else 0)
                return None

            case Instr(name, [src, dst]) if name.startswith('cmov'):
                if self.condition(name[4:]):
                    self.write(dst, self.read(src))
                return None

            case Jump(label):
                return self.target(label)

            case JumpIf(cc, label):
                return self.target(label) if self.condition(cc) else None

            case Callq(func, _):
                return self.call(func)

            case IndirectCallq(func, _):
                target = self.pc_of_address(self.read(func))
                return self.call_to(target)

            case IndirectJump(dst):
                return self.pc_of_address(self.read(dst))

            case Instr(name, _):
                raise self.fail(self.unknown_mnemonic(name))

            case _:
                raise self.fail(
                    f'{type(ins).__name__} is not an instruction this '
                    'interpreter models')

    def divide(self, divisor: int) -> None:
        if divisor == 0:
            raise self.fail('division by zero')
        # %rax and %rdx go through read()/write() like any other register, so
        # that the caller-saved bookkeeping stays in step: a cqo defines %rdx,
        # and the quotient and remainder define them again.
        dividend = self.read(Reg('rax'))
        expected = -1 if dividend < 0 else 0
        if self.read(Reg('rdx')) != expected:
            raise self.fail(
                'idivq divides %rdx:%rax, but %rdx is not the sign extension '
                'of %rax -- a cqo is missing before this')
        # x86 truncates toward zero; Python's // floors.
        quotient = abs(dividend) // abs(divisor)
        if (dividend < 0) != (divisor < 0):
            quotient = -quotient
        self.write(Reg('rax'), quotient)
        self.write(Reg('rdx'), dividend - quotient * divisor)

    # -- the loop ---------------------------------------------------------

    def run(self, max_steps: int) -> list[int]:
        steps = 0
        while not self.halted and self.pc < len(self.code):
            if steps >= max_steps:
                raise self.fail(
                    f'still running after {max_steps} instructions; '
                    'the program is probably looping')
            steps += 1
            next_pc = self.execute(self.code[self.pc])
            self.pc = self.pc + 1 if next_pc is None else next_pc
        return self.output


# ------------------------------------------------------------ entry points


def run(blocks: dict[str, list[instr]], entry: str | None = None, *,
        stdin: TextIO | None = None, stdout: TextIO | None = None,
        max_steps: int = 10_000_000, heap_size: int = 1 << 20) -> list[int]:
    """Interpret a block dictionary, returning the integers it printed."""
    if entry is None:
        entry = default_entry(blocks)
    code, labels, where = load(blocks, entry)
    machine = Machine(code, labels, where, entry,
                      stdin=stdin, stdout=stdout, heap_size=heap_size,
                      check_frames='main' in blocks)
    return machine.run(max_steps)


def interp_x86(prog: Program, **kwargs) -> list[int]:
    """Interpret a program the compilers built.  A thin adapter over run()."""
    return run(blocks_of(prog), **kwargs)


# ------------------------------------------------- driver-side conveniences
#
# The compilers all want the same two things -- a --interp that runs the
# finished program, and a --interp-after PASS that runs an intermediate one.
# They live here so that there is one implementation rather than five.

PASSES = ['select_instructions', 'assign_homes', 'allocate_registers', 'patch_instructions',
          'prelude_and_conclusion']


def add_arguments(parser) -> None:
    """Add --interp and --interp-after to a compiler's argument parser."""
    parser.add_argument('--interp', action='store_true',
                        help='interpret the compiled program instead of '
                             'writing assembly')
    parser.add_argument('--interp-after', metavar='PASS', default=None,
                        choices=PASSES,
                        help='interpret the program as it stands after PASS, '
                             'then stop: ' + ', '.join(PASSES))


def checkpoint(args, pass_name: str, prog: Program) -> None:
    """Interpret and stop, if --interp-after named this pass.

    Called after each back-end pass.  Interpreting between passes is what
    localises a miscompilation to the pass that caused it -- and it works on
    pseudo-x86, before homes exist, which gcc cannot do.
    """
    if getattr(args, 'interp_after', None) != pass_name:
        return
    print(f'# interpreting after {pass_name}', file=sys.stderr)
    interp_x86(prog)
    raise SystemExit(0)

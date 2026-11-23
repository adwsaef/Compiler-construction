import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMPILER = ROOT / "compiler" / "compile.py"
RUNTIME = ROOT / "support" / "runtime.c"
TESTS = Path(__file__).resolve().parent
REGISTER_MODES = (
    ("uncapped", ()),
    ("cap 1", ("--registers-cap", "1")),
    ("cap 8", ("--registers-cap", "8")),
)


def run(command, input_text=""):
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
    )


def check_output(name, result, expected):
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    assert result.stdout == expected, (
        f"{name} output mismatch\n"
        f"expected: {expected!r}\n"
        f"actual:   {result.stdout!r}"
    )


def run_good(source, *flags):
    input_text = source.with_suffix(".in").read_text() \
        if source.with_suffix(".in").exists() else ""
    expected = source.with_suffix(".out").read_text()

    for stage in ("type_check", "shrink", "remove_complex_operands", "explicate_control"):
        result = run([sys.executable, str(COMPILER), str(source), *flags, "--interp-after", stage], input_text)
        check_output(f"{source.name} after {stage}", result, expected)

    for mode, options in REGISTER_MODES:
        interpreted = run(
            [sys.executable, str(COMPILER), str(source), *flags, *options, "--interp"],
            input_text,
        )
        check_output(f"{source.name} {mode} interpreter", interpreted, expected)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            assembly = directory / f"{source.stem}.s"
            executable = directory / source.stem

            compiled = run([
                sys.executable, str(COMPILER), str(source), str(assembly), *flags, *options
            ])
            assert compiled.returncode == 0, (
                f"{source.name} {mode} compilation failed:\n{compiled.stderr}"
            )

            linked = run(["gcc", str(assembly), str(RUNTIME), "-o", str(executable)])
            assert linked.returncode == 0, (
                f"{source.name} {mode} linking failed:\n{linked.stderr}"
            )

            native = run([str(executable)], input_text)
            check_output(f"{source.name} {mode} native", native, expected)

        print(f"PASS {source.name} {mode}: interpreter + native")


def run_bad(source):
    function_errors = {
        "fun_name_as_value.lq": "Function name 'answer' cannot be used as a variable",
        "fun_wrong_arity.lq": "Function 'add' expects 2 arguments, got 1",
        "fun_unknown_function.lq": "Unknown function 'missing'",
    }
    if source.name in function_errors:
        for flags in ([], ["--no-type-check"]):
            result = run([sys.executable, str(COMPILER), str(source), *flags])
            assert result.returncode != 0 and function_errors[source.name] in result.stderr, result.stderr

    for mode, options in REGISTER_MODES:
        interpreted = run([
            sys.executable, str(COMPILER), str(source), *options, "--interp"
        ])
        assert interpreted.returncode != 0, (
            f"{source.name} {mode} was accepted by the interpreter"
        )

        with tempfile.TemporaryDirectory() as directory:
            assembly = Path(directory) / f"{source.stem}.s"
            compiled = run([
                sys.executable, str(COMPILER), str(source), str(assembly), *options
            ])
            assert compiled.returncode != 0, (
                f"{source.name} {mode} was accepted by the compiler"
            )

        print(f"PASS {source.name} {mode}: rejected")


def main():
    for source in sorted((TESTS / "good").glob("*.lq")):
        run_good(source)
    for source in sorted((TESTS / "untyped").glob("*.lq")):
        run_good(source, "--no-type-check")
    for source in sorted((TESTS / "bad").glob("*.lq")):
        run_bad(source)
    print("All tests passed")


if __name__ == "__main__":
    main()

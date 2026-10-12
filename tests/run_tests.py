import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMPILER = ROOT / "lqua" / "compile.py"
RUNTIME = ROOT / "support" / "runtime.c"
TESTS = Path(__file__).resolve().parent


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


def run_good(source):
    input_text = source.with_suffix(".in").read_text() \
        if source.with_suffix(".in").exists() else ""
    expected = source.with_suffix(".out").read_text()

    interpreted = run([sys.executable, str(COMPILER), str(source), "--interp"],
                      input_text)
    check_output(f"{source.name} interpreter", interpreted, expected)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        assembly = directory / f"{source.stem}.s"
        executable = directory / source.stem

        compiled = run([sys.executable, str(COMPILER), str(source), str(assembly)])
        assert compiled.returncode == 0, (
            f"{source.name} compilation failed:\n{compiled.stderr}"
        )

        linked = run(["gcc", str(assembly), str(RUNTIME), "-o", str(executable)])
        assert linked.returncode == 0, (
            f"{source.name} linking failed:\n{linked.stderr}"
        )

        native = run([str(executable)], input_text)
        check_output(f"{source.name} native", native, expected)

    print(f"PASS {source.name}: interpreter + native")


def run_bad(source):
    interpreted = run([sys.executable, str(COMPILER), str(source), "--interp"])
    assert interpreted.returncode != 0, (
        f"{source.name} was accepted by the interpreter"
    )

    with tempfile.TemporaryDirectory() as directory:
        assembly = Path(directory) / f"{source.stem}.s"
        compiled = run([sys.executable, str(COMPILER), str(source), str(assembly)])
        assert compiled.returncode != 0, (
            f"{source.name} was accepted by the compiler"
        )

    print(f"PASS {source.name}: rejected")


def main():
    for source in sorted((TESTS / "good").glob("*.lq")):
        run_good(source)
    for source in sorted((TESTS / "bad").glob("*.lq")):
        run_bad(source)
    print("All tests passed")


if __name__ == "__main__":
    main()

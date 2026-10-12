import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = Path(__file__).resolve().parent

def run_test(source, should_pass):
    input_file = source.with_suffix(".in")
    input_text = input_file.read_text() if input_file.exists() else ""

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "lqua" / "compile.py"),
            str(source),
            "--interp",
        ],
        input=input_text,
        text=True,
        capture_output=True,
    )

    if should_pass:
        assert result.returncode == 0, result.stderr

        expected = source.with_suffix(".out").read_text()
        assert result.stdout == expected, (
            f"{source.name}: output mismatch\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {result.stdout!r}"
        )
    else:
        assert result.returncode != 0, (
            f"{source.name}: expected compilation failure, but it succeeded"
        )

    status = "" if should_pass else " (rejected)"
    print(f"PASS {source.name}{status}")


for directory, should_pass in [(TEST_DIR / "good", True), (TEST_DIR / "bad", False)]:
    for source in sorted(directory.glob("*.lq")):
        run_test(source, should_pass)

print("All tests passed")

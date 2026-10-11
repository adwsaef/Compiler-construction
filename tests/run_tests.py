import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = Path(__file__).resolve().parent

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "support") + os.pathsep + env.get("PYTHONPATH", "")

for source in sorted(TEST_DIR.glob("*.lq")):
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "lqua" / "compile.py"),
                str(source),
                str(Path(tmp) / "output.s"),
                "--interp",
            ],
            input=source.with_suffix(".in").read_text(),
            text=True,
            capture_output=True,
            env=env,
        )

    assert result.returncode == 0, result.stderr
    assert result.stdout == source.with_suffix(".out").read_text(), (
        f"{source.name}: output mismatch\n"
        f"Expected: {source.with_suffix('.out').read_text()!r}\n"
        f"Actual:   {result.stdout!r}"
    )

    print(f"PASS {source.name}")

print("All tests passed")
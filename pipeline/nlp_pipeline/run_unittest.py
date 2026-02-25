"""Run unittest and write results to a file."""
import subprocess
import os
import sys

cwd = os.path.dirname(os.path.abspath(__file__))
result = subprocess.run(
    [sys.executable, "-m", "unittest", "test_vn_normalizer", "-v"],
    capture_output=True,
    text=True,
    cwd=cwd,
    encoding="utf-8",
    errors="replace"
)

outpath = os.path.join(cwd, "unittest_results.txt")
with open(outpath, "w", encoding="utf-8") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)
    f.write("\nSTDERR:\n")
    f.write(result.stderr)
    f.write(f"\nReturn code: {result.returncode}\n")

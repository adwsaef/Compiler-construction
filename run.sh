#!/usr/bin/env bash
set -e

source_file=${1:?Usage: $0 file.lq}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build=$(mktemp -d)
trap 'rm -rf -- "$build"' EXIT

PYTHONPATH="$root/support" python3 "$root/compiler/compile.py" -- "$source_file" "$build/program.s"
gcc "$build/program.s" "$root/support/runtime.c" -o "$build/program"
"$build/program"

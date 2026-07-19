#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 OUTPUT ANNOTATION -- COMMAND [ARG ...]" >&2
    exit 2
}

[[ $# -ge 4 ]] || usage
output=$1
annotation=$2
shift 2
[[ $1 == "--" ]] || usage
shift
[[ $# -gt 0 ]] || usage

for tool in valgrind cg_merge cg_annotate; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "cachegrind: required tool is missing: $tool" >&2
        exit 2
    fi
done

mkdir -p "$(dirname "$output")" "$(dirname "$annotation")"
raw_dir="${output}.d"
mkdir -p "$raw_dir"
rm -f "$raw_dir"/cachegrind.*.out "$output" "$annotation"

valgrind \
    --tool=cachegrind \
    --cache-sim=yes \
    --branch-sim=no \
    --trace-children=yes \
    --cachegrind-out-file="$raw_dir/cachegrind.%p.out" \
    -- "$@"

shopt -s nullglob
raw_files=("$raw_dir"/cachegrind.*.out)
shopt -u nullglob
if [[ ${#raw_files[@]} -eq 0 ]]; then
    echo "cachegrind: no per-process reports were produced in $raw_dir" >&2
    exit 1
fi

cg_merge -o "$output" "${raw_files[@]}"
cg_annotate "$output" > "$annotation"

echo "cachegrind: merged ${#raw_files[@]} process report(s) into $output"
echo "cachegrind: annotation written to $annotation"

#!/usr/bin/env bash
# Profile CK v8 and llama.cpp prefill with Intel Advisor/VTune.
#
# Default target is Qwen3.5 0.8B Q4_K_M because CK's v8 op profiler shows
# prefill time is dominated by mlp_gate_up -> gemm_nt_q4_k_q8_k on this model.
#
# Examples:
#   scripts/profile_v8_prefill_roofline.sh --tool advisor --engine both --prompt 128
#   scripts/profile_v8_prefill_roofline.sh --tool vtune-hotspots --engine ck --prompt 512
#   scripts/profile_v8_prefill_roofline.sh --tool all --engine both --prompt 128

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/profile_results/v8_prefill_roofline}"
STAMP="$(date +%Y%m%d-%H%M%S)"

ENGINE="both"
TOOL="advisor"
PROMPT_TOKENS="${CK_PROFILE_PROMPT_TOKENS:-128}"
DECODE_TOKENS="${CK_PROFILE_DECODE_TOKENS:-1}"
THREADS="${CK_NUM_THREADS:-12}"
TOKEN_ID="${CK_PROFILE_TOKEN_ID:-100}"
MODEL_KEY="qwen35"

QWEN35_DIR="${CK_QWEN35_DIR:-$HOME/.cache/ck-engine-v8/models/unsloth--Qwen3.5-0.8B-GGUF}"
CK_CLI="${CK_CLI:-$ROOT_DIR/build/ck-cli-v8}"
CK_LIB="${CK_LIB:-$QWEN35_DIR/libmodel.so}"
CK_WEIGHTS="${CK_WEIGHTS:-$QWEN35_DIR/weights.bump}"
GGUF="${GGUF:-$QWEN35_DIR/Qwen3.5-0.8B-Q4_K_M.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT_DIR/llama.cpp/build/bin/llama-bench}"

usage() {
    sed -n '1,24p' "$0"
    echo
    echo "Options:"
    echo "  --engine ck|llama|both        default: $ENGINE"
    echo "  --tool advisor|vtune-hotspots|vtune-memory|vtune-uarch|all"
    echo "                                default: $TOOL"
    echo "  --prompt N                    prompt tokens, default: $PROMPT_TOKENS"
    echo "  --decode N                    decode tokens, default: $DECODE_TOKENS"
    echo "  --threads N                   CPU threads, default: $THREADS"
    echo "  --token-id N                  repeated CK prompt token id, default: $TOKEN_ID"
    echo "  --out DIR                     output root, default: $OUT_ROOT"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --engine) ENGINE="$2"; shift 2 ;;
        --tool) TOOL="$2"; shift 2 ;;
        --prompt) PROMPT_TOKENS="$2"; shift 2 ;;
        --decode) DECODE_TOKENS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --token-id) TOKEN_ID="$2"; shift 2 ;;
        --out) OUT_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "$ENGINE" in
    ck|llama|both) ;;
    *) echo "invalid --engine: $ENGINE" >&2; exit 2 ;;
esac

case "$TOOL" in
    advisor|vtune-hotspots|vtune-memory|vtune-uarch|all) ;;
    *) echo "invalid --tool: $TOOL" >&2; exit 2 ;;
esac

require_file() {
    if [ ! -e "$1" ]; then
        echo "missing required file: $1" >&2
        exit 1
    fi
}

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required tool: $1" >&2
        echo "try: source /opt/intel/oneapi/setvars.sh" >&2
        exit 1
    fi
}

repeat_token_csv() {
    local n="$1"
    local tok="$2"
    local out="$tok"
    local i=1
    while [ "$i" -lt "$n" ]; do
        out="$out,$tok"
        i=$((i + 1))
    done
    printf '%s' "$out"
}

require_file "$CK_CLI"
require_file "$CK_LIB"
require_file "$CK_WEIGHTS"
require_file "$GGUF"
require_file "$LLAMA_BENCH"

if [ "$TOOL" = "advisor" ] || [ "$TOOL" = "all" ]; then
    require_tool advisor
fi
if [ "$TOOL" = "vtune-hotspots" ] || [ "$TOOL" = "vtune-memory" ] || [ "$TOOL" = "vtune-uarch" ] || [ "$TOOL" = "all" ]; then
    require_tool vtune
fi

RUN_ROOT="$OUT_ROOT/$STAMP"
mkdir -p "$RUN_ROOT"

TOKEN_CSV="$(repeat_token_csv "$PROMPT_TOKENS" "$TOKEN_ID")"
CONTEXT=$((PROMPT_TOKENS + DECODE_TOKENS + 16))
CK_DECODE=$((DECODE_TOKENS + 1))

export CK_NUM_THREADS="$THREADS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LD_LIBRARY_PATH="$ROOT_DIR/build:$QWEN35_DIR:${LD_LIBRARY_PATH:-}"

CK_CMD=(
    "$CK_CLI" "$CK_LIB" "$CK_WEIGHTS"
    --prompt-tokens "$TOKEN_CSV"
    --max-tokens "$CK_DECODE"
    --context "$CONTEXT"
    --temperature 0
    --ignore-eos
    --quiet-output
    --no-chat-template
    --no-stream
    --timing
)

LLAMA_CMD=(
    "$LLAMA_BENCH"
    -m "$GGUF"
    -p "$PROMPT_TOKENS"
    -n "$DECODE_TOKENS"
    -t "$THREADS"
    -ngl 0
    -r 1
    -o json
)

write_command_file() {
    {
        echo "model_key=$MODEL_KEY"
        echo "threads=$THREADS"
        echo "prompt_tokens=$PROMPT_TOKENS"
        echo "decode_tokens=$DECODE_TOKENS"
        echo "ck_cli=$CK_CLI"
        echo "ck_lib=$CK_LIB"
        echo "ck_weights=$CK_WEIGHTS"
        echo "gguf=$GGUF"
        echo
        printf 'CK_CMD='
        printf '%q ' "${CK_CMD[@]}"
        echo
        printf 'LLAMA_CMD='
        printf '%q ' "${LLAMA_CMD[@]}"
        echo
    } > "$RUN_ROOT/commands.txt"
}

run_advisor_one() {
    local name="$1"
    shift
    local project="$RUN_ROOT/advisor_${name}"
    echo "Advisor roofline: $name -> $project"
    advisor --collect=roofline \
        --project-dir="$project" \
        --search-dir "src:r=$ROOT_DIR" \
        -- "$@"
    advisor --report=roofline --project-dir="$project" \
        --report-output="$RUN_ROOT/advisor_${name}_roofline.txt" \
        --format=text >/dev/null 2>&1 || true
    advisor --report=survey --project-dir="$project" \
        --report-output="$RUN_ROOT/advisor_${name}_survey.txt" \
        --format=text >/dev/null 2>&1 || true
}

run_vtune_one() {
    local kind="$1"
    local name="$2"
    shift 2
    local result="$RUN_ROOT/vtune_${kind}_${name}"
    local collect="$kind"
    if [ "$kind" = "vtune-hotspots" ]; then
        collect="hotspots"
    elif [ "$kind" = "vtune-memory" ]; then
        collect="memory-access"
    elif [ "$kind" = "vtune-uarch" ]; then
        collect="uarch-exploration"
    fi
    echo "VTune $collect: $name -> $result"
    vtune -collect "$collect" -result-dir "$result" -quiet -- "$@"
    vtune -report summary -result-dir "$result" \
        -report-output="$RUN_ROOT/${kind}_${name}_summary.txt" \
        -format=text >/dev/null 2>&1 || true
    vtune -report hotspots -result-dir "$result" \
        -report-output="$RUN_ROOT/${kind}_${name}_hotspots.txt" \
        -format=text >/dev/null 2>&1 || true
}

run_for_engine() {
    local which="$1"
    local cmd_ref="$2"
    shift 2
    if [ "$TOOL" = "advisor" ] || [ "$TOOL" = "all" ]; then
        run_advisor_one "$which" "$@"
    fi
    if [ "$TOOL" = "vtune-hotspots" ] || [ "$TOOL" = "all" ]; then
        run_vtune_one "vtune-hotspots" "$which" "$@"
    fi
    if [ "$TOOL" = "vtune-memory" ] || [ "$TOOL" = "all" ]; then
        run_vtune_one "vtune-memory" "$which" "$@"
    fi
    if [ "$TOOL" = "vtune-uarch" ] || [ "$TOOL" = "all" ]; then
        run_vtune_one "vtune-uarch" "$which" "$@"
    fi
}

write_command_file

echo "Output: $RUN_ROOT"
echo "Threads: $THREADS, prompt=$PROMPT_TOKENS, decode=$DECODE_TOKENS"

if [ "$ENGINE" = "ck" ] || [ "$ENGINE" = "both" ]; then
    run_for_engine "ck" CK_CMD "${CK_CMD[@]}"
fi

if [ "$ENGINE" = "llama" ] || [ "$ENGINE" = "both" ]; then
    run_for_engine "llama" LLAMA_CMD "${LLAMA_CMD[@]}"
fi

echo
echo "Done. Commands and reports: $RUN_ROOT"

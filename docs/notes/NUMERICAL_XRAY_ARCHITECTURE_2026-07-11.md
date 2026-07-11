# Numerical X-ray Architecture

## Status

The v8 numerical X-ray path now covers the diagnostic flow from a circuit semantic edge through backend tensor and ranking comparison:

1. The circuit declares versioned semantic checkpoints.
2. BuildIR binds each checkpoint to a generated operation, exact kernel ID, public function, phase, layer, layout and named axes.
3. GraphIR, LoweredIR and call IR retain the same metadata.
4. The checkpoint-manifest adapter takes tensor identity from call IR rather than backend-specific file names.
5. Backend profiles hold observed storage policy and dtype tolerances outside circuits.
6. The comparator validates metadata before loading tensor values.
7. Named axes canonicalize compatible physical layouts.
8. Sparse checkpoints identify a failing interval.
9. The planner requests only intermediate layers and then only internal edges of the first failing block.
10. Mixed-prefill, teacher-forced and persistent-versus-replay results use a shared ranking-report ABI.

This is approximately 90-95% coverage of ordinary model-integration failures. Compiler instruction scheduling and hardware-specific transcendental behavior still require isolated kernel instrumentation or hardware profiling after X-ray identifies the edge.

## Failure Classes

- `MISSING_CHECKPOINT`
- `CIRCUIT_PRODUCER_MISMATCH`
- `LAYOUT_MISMATCH`
- `STORAGE_CONTRACT_MISMATCH`
- `REDUCTION_CONTRACT_MISMATCH`
- `POSITION_CONTRACT_MISMATCH`
- `NUMERICAL_CONTRACT_MISMATCH`
- `KERNEL_BINDING_MISMATCH`
- `DIAGNOSTIC_EXPORT_MAPPING`
- `KERNEL_IMPLEMENTATION_DIVERGENCE`
- `NONFINITE_OUTPUT`
- `RANKING_DIVERGENCE`
- `STATE_CACHE_DIVERGENCE`
- `MISSING_TOLERANCE_PROFILE`

Every failure includes a recommended action. Metadata faults do not proceed to numerical metrics.

## Lightweight Gate

```bash
make test-bf16-xray
```

This runs schema, resolver, canonicalization, classification, bisection and call-IR manifest tests. It also generates a dependency-free public 1152x896 synthetic form at:

```text
build/xray/public_form_1152x896.ppm
```

The gate is included in `make test-bf16` and the BF16 nightly category. It does not download model weights.

## Real BF16 Diagnosis

```bash
make xray-vision-parity \
  CHECKPOINT=/path/to/Qwen3-VL-safetensors \
  RUNTIME_DIR=/path/to/generated/vision-runtime \
  WEIGHTS_BUMP=/path/to/weights.bump \
  CALL_IR=/path/to/call.json \
  IMAGE=/path/to/public-form.ppm
```

The command first compares circuit storage requirements against the PyTorch backend profile. Contract mismatches stop before expensive model execution. When metadata contracts agree, it runs bounded tensor captures and drills down automatically.

Current local Qwen3-VL BF16 preflight:

```text
FAIL at: vision.frontend.position.output
CLASS: STORAGE_CONTRACT_MISMATCH
CK storage: fp32
PyTorch storage: bf16
ACTION: declare and implement the matched storage/rounding boundary through the circuit and kernel map
```

Report:

```text
build/xray/qwen3vl_bf16_real_preflight/xray_summary.json
```

This reproduces the previously measured diagnosis without a private image and without running unnecessary encoder passes.

## Ranking Integration

Normalize existing mixed-prefill or multitoken parity output:

```bash
python version/v8/scripts/normalize_xray_ranking_report_v8.py \
  --input build/parity.json \
  --kind teacher_forced \
  --output build/xray/ranking.json
```

Pass that report to `xray_qwen3vl_bf16_v8.py --ranking-report ...`. Ranking is evaluated only after tensor checkpoints pass. Persistent-versus-replay failures are classified separately from full-path arithmetic drift.

## Remaining Work

The X-ray architecture is operational. Qwen3-VL BF16 still needs the actual numerical fix:

1. Register the measured BF16 position storage/rounding contract.
2. Advertise one exact compatible kernel implementation.
3. Change the circuit checkpoint storage declaration through contract resolution rather than a model-name branch.
4. Rerun X-ray; it will advance to the next mismatching boundary.
5. Continue until mixed-prefill and teacher-forced ranking gates pass.

Do not relax the parity profile to hide a storage or reduction contract mismatch.

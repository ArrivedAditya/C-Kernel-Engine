# v8 Generated Runtime Ownership Audit

Date: 2026-08-01

Audited commit: `93e5f54c7`

## Objective

The intended v8 architecture is:

```text
model metadata + circuit + kernel maps
                  |
                  v
       strict mechanical compiler
                  |
                  v
       generated native runtime
                  |
                  v
        thin C CLI / Rust server
```

The generated runtime should own model execution semantics. A host should only
load artifacts, provide input bytes, stream output bytes, and manage transport.
It must not know that a model is Qwen, Gemma, Whisper, GLM, Kimi, or any other
family.

This audit answers two questions:

1. Which model or modality semantics are still hardcoded outside circuits and
   kernel maps?
2. Can `ck_cli_v8.c` become a common text, vision, and audio runtime with only a
   small generic ABI?

The short answer is: **yes, incrementally**. The generated model owns tensor
execution, chat/stop metadata, the complete Whisper audio frontend, and the
first versioned capability ABI. Python still owns raw-image preprocessing,
long-audio/timestamp scheduling, and some multimodal composition rules.

## Implemented During This Audit

The first common runtime ABI is now generated from resolved artifacts rather
than model names. `CKModelRuntimeDescriptorV8` declares artifact role,
capabilities, context/vocabulary dimensions, encoder-output geometry, and
decoder encoder-memory geometry. The C host validates ABI version, structure
size, known capability bits, role consistency, shape pairs, and every required
export before execution.

The generated runtime now owns:

- single-turn chat formatting from the circuit `chat_contract`;
- stop-token identity from generated tokenizer metadata;
- Whisper decoder prefix and logits policy from `generation_config.json`;
- generic encoder-output discovery;
- declared WAV input, normalized image-tensor input, mixed prefill, and
  encoder-memory capabilities.

The native CLI now runs Whisper Base without Python in the inference process:

```text
WAV bytes -> generated WAV/frontend/encoder -> generated encoder output
          -> generated decoder memory bind -> generated prefix/policy
          -> persistent decode -> generated tokenizer output
```

The JFK fixture produced the expected prefix, "And so my fellow Americans, ask
not what your country can do for you," through this path. This run also exposed
and fixed a host bug where generated stop-token metadata accidentally bypassed
all token output. Descriptor and output behavior are covered by regressions.

Boundaries that remain deliberately fail-closed:

- Native timestamp generation is not advertised as complete. The generated
  policy rejects timestamp mode until timestamp filtering and seek updates are
  fully native.
- Vision encoders advertise normalized FP32 image-tensor input, not raw image
  bytes. JPEG/PNG decode, resize, normalization, dynamic geometry, marker
  insertion, and segmented scheduling still need circuit providers before a
  `RAW_IMAGE_ENCODER` capability can be emitted.
- `--bridge-report` remains the native vision compatibility route for a
  previously captured, provenance-bearing multimodal prefix.
- Family-name chat tables remain only for old artifacts with no v8 descriptor.
  New artifacts without a generated chat formatter fail unless the caller
  explicitly requests a raw prompt.

## Architectural Rule

The existing design documents already state the right standard:

- Circuits declare operations, graph edges, branches, state transitions, and
  required numerical semantics.
- Kernel maps declare concrete providers, call ABI, dtype/layout constraints,
  numerical behavior, ISA, threading, scratch, and implementation sources.
- Canonical model metadata declares checkpoint-specific dimensions, token IDs,
  chat templates, preprocessing constants, and weight bindings.
- The compiler parses, resolves exactly one provider, validates, plans memory,
  and emits calls. It must not infer behavior from model-family names.
- A host CLI or server owns files, sockets, queues, cancellation, and output I/O.

This is consistent with `version/v8/circuits/README.md:47` and
`version/v8/circuits/PIPELINE.md:5`. The implementation has not fully reached
that standard.

## Current Evidence

The current architecture dashboard reports:

- 12 promoted templates.
- 35 explicit critical tensor edges.
- 78 implicit or missing critical edges.
- 10 of 12 promoted templates have warnings.
- Only GLM4 and Kimi-VL have complete critical-edge coverage in that dashboard.

The kernel-map inventory contains 272 map files:

- 105 declare `call_abi`.
- 86 declare a numerical contract or numerical capabilities.

The circuit inventory contains 18 circuits:

- 94 required numerical contracts in total.
- No circuit currently populates the available top-level `runtime_defaults` or
  `weight_policy` fields.
- The runtime-default schema is narrowly limited to a few quantization and MLP
  preferences. It cannot describe modality capabilities, preprocessing,
  generation, or execution schedules.

At audited base `93e5f54c7`, code generation emitted 44 literal
`ck_model_*` symbols. The native CLI resolved 33 symbols and left 17 generated
capabilities unconsumed, including audio entrypoints, encoder-memory binding,
generated stop-token APIs, and generated BOS/EOS APIs. The sets are not exact
subsets because the CLI also resolves compatibility symbols emitted outside
the scanned code generators.

These counts are not all defects. They show that the declarative path exists,
but coverage and runtime composition are incomplete.

## Native CLI History

`git log --follow` and `git blame` explain why the native CLI predates the
current architecture:

- The chat-template table, model-name detection, loader, and tokenizer host
  logic were introduced by `5481e88e5` on 2026-03-30.
- Multimodal exports were appended in April 2026.
- Gemma was appended by `829c708f0` on 2026-05-09.
- Gemma4 was appended by `ec7ec358b` on 2026-06-12.
- The latest commit touching the file before this audit was `74b95d613` on
  2026-06-21.

That implementation was reasonable as an early native bootstrap. It predates
the current circuit chat contracts, generated stop-token API, audio circuits,
Whisper runtime, and strict numerical-provider work. Continuing to append
families to the same table is no longer compatible with v8 ownership.

The first hardening increment accompanying this audit is additive:

- New generated artifacts export `ck_model_has_chat_template` and
  `ck_model_format_chat` from their circuit `chat_contract`.
- `ck_cli_v8.c` resolves and prefers that generated formatter.
- The CLI resolves and prefers generated stop-token/BOS/EOS APIs.
- Family-name templates and textual EOS probing remain compatibility fallbacks
  for old cached artifacts only.

After this increment, the CLI resolves 40 generated or compatibility symbols.
The audit still finds 12 generated symbols that it does not consume, primarily
audio, encoder-memory, and diagnostic entrypoints. Generated EOS metadata is
now authoritative, but resolving every textual `token_stop_marker` into a
canonical token ID remains part of the versioned session-ABI work.

This does not complete the common session ABI or modality work below. It stops
newly generated text runtimes from depending on the most consequential legacy
guesses without invalidating existing caches.

## Classification

Not every conditional in a compiler is hardcoding.

### Correct compiler mechanics

These belong in generic compiler code:

- JSON/schema validation.
- Symbolic graph construction.
- Unique provider resolution.
- Shape and dtype checking.
- Memory lifetime planning.
- Generic loop, branch, collect, stitch, route, dispatch, and combine emission.
- Generic call emission from a resolved `call_abi`.
- Capability-table and ABI generation.

### Acceptable ingestion specialization

GGUF and safetensors converters may understand source format and architecture
metadata. Their output must be canonical. Runtime lowering and code generation
must not repeat those family decisions.

### Incorrect runtime/compiler hardcoding

The following must be declarative:

- Checking model names to choose tokenization or chat formatting.
- Checking model names to choose image geometry or preprocessing.
- Checking model names or op names to choose concrete numerical providers.
- Model-specific environment flags that change production execution.
- Python-owned forced prefixes, token suppression, timestamp constraints,
  stopping, or sampling.
- Python-owned encoder/decoder composition.
- Architecture-named operations where generic graph primitives suffice.

## Findings

### P0: The native CLI hardcodes chat families

`version/v8/src/ck_cli_v8.c:153` declares Qwen, Llama, ChatML, Mistral, Gemma,
and Gemma4 template enums and literal strings. `ck_cli_v8.c:1679` infers the
template by searching the model name. `ck_cli_v8.c:1696` has Gemma-specific
formatting behavior.

This is precisely the behavior that caused the earlier GLM formatting failure:
a numerically usable model can still generate repeated or incoherent output
when the host guesses the prompt contract incorrectly.

Correct owner:

- Canonical tokenizer/chat metadata stores the source template and token IDs.
- A `prompt_template_apply` provider executes the resolved template contract.
- The circuit places that operation before tokenization.
- The CLI passes role/content messages without model-name detection.

### P0: Whisper generation remains Python-owned

The generated encoder correctly exposes `ck_model_run_audio_wav_window`, and
the audio circuit explicitly contains WAV decode, resampling, padding, STFT,
mel filters, log-Mel, feature windowing, Conv1D, attention, and encoder output.
This is the strongest current example of the intended architecture.

However, `version/v8/scripts/run_whisper_v8.py` still owns:

- Forced decoder-prefix construction at line 280.
- Long-audio window planning at line 303.
- Timestamp-to-global-time conversion at line 337.
- Timestamp seek advancement at line 355.
- Timestamp logits filtering at line 381.
- Encoder-memory binding and decoder composition at line 436.
- Suppressed-token and begin-suppressed-token masks at line 498.
- Greedy selection and EOS handling at line 508.
- Python tokenizer decoding at line 535.

The audio decoder circuit describes several of these policies as text in
`contract.logits_contract.generation_policies`, but descriptions are not
executable graph operations. Python remains the real implementation.

Correct owner:

```text
audio_window_plan
audio_frontend
audio_encoder
encoder_memory_bind
decoder_prefix_build
decoder_prefill
logits_mask
timestamp_logits_filter
sample
stop_check
tokenizer_decode
timestamp_seek_update
```

Each item should be a circuit operation. Kernel maps should select its native
provider. The generated schedule should invoke them in order.

### P0: Vision preprocessing and geometry remain model-specific Python

`version/v8/scripts/run_multimodal_bridge_v8.py` owns significant model
semantics:

- It guesses output activation names such as `vision_output` and
  `vision_bridge_output` at line 682.
- It parses literal ChatML and vision marker strings at line 1053.
- It implements Qwen3-VL smart-resize geometry in
  `_qwen3vl_geometry_overrides` at line 1612.
- It implements Gemma4 geometry in `_gemma4_geometry_overrides` at line 1679.
- It branches on `qwen3_vl_vision` and `gemma4_vision` at line 1974.
- It performs image decode, resize, normalization, layout conversion, buffer
  copying, output discovery, and prefix-grid derivation around line 2480.

Correct owner:

```text
image_decode
image_geometry_select
image_resize
image_normalize
image_layout_transform
patchify
vision_encoder
vision_projector
encoder_output_export
prompt_marker_insert
multimodal_position_build
segmented_prefill
```

Geometry algorithms may have different providers. The circuit selects a
semantic contract; the host does not branch on a model name.

### P0: There is no common generated capability/session ABI

The native CLI currently resolves individual historical exports in
`load_model_api`. This increment adds generated chat and stop metadata, but it
still does not resolve generated audio entrypoints or encoder-memory binding.

Generated but unconsumed runtime capabilities include:

- `ck_model_run_audio_wav`
- `ck_model_run_audio_wav_window`
- `ck_model_prepare_audio_wav_window`
- `ck_model_run_encoder`
- `ck_model_set_encoder_memory`

The generated BOS/EOS and stop-token exports are now consumed directly. They
remain listed in the proposed common ABI below because adding each optional
symbol independently is transitional and will not scale.

Adding every new optional symbol directly to `ModelAPI` will not scale. The
generated runtime needs one versioned capability API.

Recommended ABI:

```c
uint32_t ck_model_abi_version(void);
uint64_t ck_model_get_capabilities(void);
int ck_model_create_session(const CKSessionOptions *, CKSession **);
int ck_model_bind_input(CKSession *, const CKInput *);
int ck_model_generate_begin(CKSession *, const CKMessage *, size_t);
int ck_model_generate_next(CKSession *, CKOutputEvent *);
void ck_model_cancel(CKSession *);
void ck_model_destroy_session(CKSession *);
```

`CKInput` is tagged (`text`, `image_bytes`, `audio_bytes`, future modalities).
`CKOutputEvent` is tagged (`text_bytes`, `token`, `timestamp`, `completed`,
`error`). The generated runtime owns tokenization and detokenization.

The CLI then needs no Qwen, Gemma, Whisper, or vision code. Unsupported input
can produce a capability warning and continue with text; `--strict-inputs`
should make that a hard error.

### P1: The IR builder still contains model-family decisions

`version/v8/scripts/build_ir_v8.py` is a monolithic parser, graph builder,
resolver, and policy engine. The documented `op_builders_v8.py` evolution lane
does not exist yet.

Concrete examples:

- Gemma tokenizer behavior is inferred from model/template names at line 1973.
- A Gemma FP32-logits guard remains runtime configuration at line 5590.
- MLA BF16/F32 provider IDs are selected directly at line 6071.
- Gemma4-specific providers are selected directly at line 6165.
- Qwen3.5-specific quantization insertion rationale appears at line 6681.
- Additional architecture-specific weight fallbacks, activation routing, patch
  sequencing, and M-RoPE parameter injection occur later in the same file.

Some dtype-based resolution is valid. The defect is returning a concrete
provider from architecture knowledge rather than asking the registry for the
unique provider satisfying the circuit contract, tensor dtypes, phase, shape,
and ISA.

Correct owner:

- Tokenizer mode: canonical tokenizer metadata.
- Logits precision: required circuit numerical contract.
- MLA/Gemma/DeltaNet provider selection: kernel-map `provides`, `requires`, and
  numerical capabilities.
- Weight aliases: circuit `weight_policy.op_bindings` or canonical manifest
  roles.
- Quantization boundaries: explicit `quantize`/`dequantize` graph edges with
  input/output dtype contracts.

### P1: Code generation duplicates operation ABI knowledge

Only 105 of 272 kernel maps currently declare `call_abi`. Consequently,
`codegen_core_v8.py` and `codegen_prefill_v8.py` contain long op/function
conditional chains that know argument order, runtime state, dimensions, and
special output handling.

Examples include Gemma4-specific attention selection at
`codegen_prefill_v8.py:942`, architecture-named operations at lines 1135 and
1398, and model-specific diagnostic modes in `codegen_core_v8.py:2348`.

The target is not zero op-specific code. Generic control operations require
emitters. The target is:

- Ordinary leaf calls are emitted entirely from resolved `call_abi`.
- Generic control primitives have generic emitters.
- Debug exports come from checkpoint metadata, not op-name chains.
- No emitter selects a replacement provider.
- No emitter contains a model-family name.

### P1: Audio entrypoint generation is still a specialized post-pass

`version/v8/scripts/codegen_v8.py:61` finds a fixed set of audio op names and
hand-emits the audio schedule. It is generated C and uses resolved call IR,
which is substantially better than Python selecting leaf kernels. It is still
a modality-specific codegen pass with hardcoded branching and error ordering.

Similarly, multimodal bridge APIs are injected after core code generation at
`codegen_v8.py:1029`.

Correct owner:

- Add a generic `entrypoints` or `execution_schedules` section to the circuit.
- A schedule declares ordered operations, conditions, state transitions,
  errors, and exported input/output types.
- Codegen mechanically emits any declared entrypoint.
- Audio and vision become data supplied to the same schedule emitter.

### P1: Circuit contract coverage is incomplete

The architecture dashboard reports 78 implicit/missing critical edges. Qwen2,
Qwen3, Qwen3-VL, Gemma3, Gemma4, Gemma4-Vision, Nemotron-H, and Llama currently
have no explicit critical-edge count in that dashboard.

This permits the builder to infer producer/consumer wiring and makes it harder
to prove that a new model is composition rather than another hidden branch.

Promotion should require:

- Every critical tensor input has an explicit producer.
- Every state buffer declares initialization, update, persistence, capacity,
  semantic extent, physical extent, and invalidation.
- Every numerically sensitive operation resolves one required contract.
- Every public entrypoint declares its input/output capability contract.

### P2: Production behavior is controlled by model-specific environment flags

Examples include `CK_QWEN3VL_DISABLE_PREFILL_DEEPSTACK` and Qwen3-VL OCR
profile-dependent fusion selection in `codegen_prefill_v8.py`.

Debug flags are acceptable when they only capture evidence. A model-specific
flag that changes the production graph or provider is not. Production choices
belong in resolved circuit/kernel-map policy and must appear in artifact
provenance.

### P2: Diagnostic identity is partly derived from op names

Codegen contains architecture-specific X-Ray labels and dump modes. Diagnostic
names should come from circuit checkpoint IDs and resolved execution identity.
This keeps X-Ray aligned with the same declarative source as execution.

## Ownership Matrix

| Concern | Correct owner |
|---|---|
| Checkpoint dimensions and token IDs | canonical model metadata |
| Chat template and role rules | canonical metadata + `prompt_template_apply` provider |
| Tensor graph and operation order | circuit |
| Branches, recurrent state, cache transitions | circuit |
| Input modalities and public entrypoints | circuit capability/schedule declarations |
| Preprocessing constants and geometry policy | canonical metadata + circuit params |
| Concrete function, source, ISA, threading | kernel map |
| Input/output dtype, layout, reduction order | kernel map numerical contract |
| Native call argument binding | kernel-map `call_abi` |
| Weight semantic roles | canonical manifest + circuit weight policy |
| Parse, validate, resolve, plan, emit | compiler |
| Files, terminal, HTTP, queues, cancellation | C CLI / Rust server |

## Required Schema Additions

The current `runtime_defaults` schema is too narrow. Prefer explicit top-level
sections rather than accumulating more model knobs:

```json
{
  "capabilities": {
    "inputs": ["text", "image_bytes", "audio_bytes"],
    "outputs": ["text_bytes", "timestamps"],
    "streaming": true
  },
  "entrypoints": {
    "generate": {
      "inputs": ["messages", "optional_modalities"],
      "schedule": [
        "prompt_template_apply",
        "tokenizer_encode",
        "modality_preprocess",
        "encoder_execute",
        "encoder_decoder_bind",
        "prefill",
        "generation_loop"
      ]
    }
  },
  "generation": {
    "prefix_policy": "metadata_driven",
    "logits_processors": [],
    "sampler": "greedy",
    "stop_policy": "model_stop_tokens",
    "detokenizer": "model_tokenizer"
  }
}
```

Names in this example are semantic operation IDs, not C functions. Kernel maps
resolve each ID to a provider.

## Migration Plan

### Phase 0: Prevent new hardcoding

1. Add a static audit that records architecture-name references in
   `build_ir_v8.py`, `codegen_*_v8.py`, and `ck_cli_v8.c`.
2. Store a reviewed baseline and fail CI on new references.
3. Shrink the allowlist as each case is migrated.
4. Require every new kernel map to declare `call_abi` and numerical semantics.

### Phase 1: Add the common capability/session ABI

1. Generate an ABI version and capability bitset/table.
2. Generate tagged input and output-event APIs.
3. Make `ck_cli_v8.c` use only this API for inference.
4. Retain old exports temporarily behind an adapter.
5. Delete model-name chat-template detection from the CLI.

Completion gate: the same unmodified native CLI runs a text model and reports
unsupported image/audio inputs from generated capabilities.

### Phase 2: Move Whisper generation into the generated runtime

1. Convert forced prefix, logits masks, timestamps, sampling, stop, and decode
   into executable circuit operations.
2. Add kernel maps and native providers for each operation.
3. Generate the encoder-memory bind and generation loop schedule.
4. Use the generated tokenizer for final output.
5. Reduce `run_whisper_v8.py` to artifact preparation and parity/benchmark
   orchestration.

Completion gate: `ck_cli_v8 --audio file.wav` transcribes Tiny/Base without
Python in the inference process and matches the existing transcript/timestamp
oracles.

### Phase 3: Move vision preprocessing and bridge scheduling

1. Add image-byte input and image decode providers.
2. Move Qwen/Gemma geometry into semantic geometry contracts.
3. Move resize, normalize, layout, patchify, marker insertion, M-RoPE position
   construction, and segmented prefill into circuit schedules.
4. Export encoder output through declared output IDs, never guessed names.

Completion gate: `ck_cli_v8 --image file.jpg --prompt ...` runs Qwen3-VL and
Gemma4 with no model-family branch in the CLI or bridge host.

### Phase 4: Make call-ABI emission generic

1. Add `call_abi` to every promoted leaf provider.
2. Generate calls from resolved arguments.
3. Replace op-name debug chains with circuit checkpoint declarations.
4. Keep only generic control-flow emitters.

Completion gate: adding a new leaf provider requires a kernel map and C source,
not a codegen conditional.

### Phase 5: Remove builder family policy

1. Move tokenizer overrides to canonical conversion metadata.
2. Move weight fallbacks to manifest roles and circuit weight policy.
3. Make quantization boundaries explicit graph operations.
4. Replace direct provider returns with contract queries.
5. Introduce the documented symbolic op-builder layer or equivalent generic
   graph construction.

Completion gate: the compiler source has no promoted model-family names.

### Phase 6: Attach the Rust server

The server should call the same session ABI as the native CLI. It owns HTTP,
stream framing, bounded queues, cancellation, and concurrency. It must not own
tokenization, prompt formatting, sampling, modality preprocessing, or model
execution schedules.

## Regression Gates

The migration is complete only with all of these gates:

1. **No-new-family-branch gate**: compiler and native CLI architecture-name
   baseline can only decrease.
2. **Circuit completeness gate**: zero missing critical edges for promoted
   circuits.
3. **Provider uniqueness gate**: every semantic operation resolves exactly one
   provider for the artifact target.
4. **Call-ABI gate**: every promoted leaf provider has a validated `call_abi`.
5. **Capability/ABI gate**: generated capability declarations match exported
   symbols and the CLI consumes the declared version.
6. **Unsupported-input gate**: capability mismatch warns or fails under
   `--strict-inputs`; it never crashes or silently invokes a guessed path.
7. **Python-independence gate**: native text, image, and audio E2E tests execute
   without Python after artifacts are built.
8. **Parity gate**: native and Python orchestration produce identical tokens,
   timestamps, stop reasons, and declared preprocessing hashes.
9. **Provenance gate**: circuit, kernel maps, registry, generated C, compiler,
   flags, runtime, tokenizer, and model hashes are recorded.

## Final Assessment

The architecture is viable and parts of it are already strong:

- Tensor execution is generated.
- Numerical contracts and kernel maps can represent exact provider semantics.
- Whisper demonstrates that even WAV parsing through encoder execution can be
  circuit-driven native code.
- Qwen3-VL demonstrates that complex segmented prefill and multimodal positions
  can be generated.

The remaining problem is ownership, not feasibility. The project currently has
two competing runtime layers: generated C owns tensor math, while Python and
`ck_cli_v8.c` still own portions of model semantics. The correct next step is
not to add more family handling to either host. It is to make generation,
capabilities, preprocessing schedules, and public entrypoints declarative, then
reduce both hosts to clients of one generated session ABI.

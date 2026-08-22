# State Model Interface (SMI)

**State Model Interface (SMI)** is a minimal protocol for structuring causal streams between stateful models and their environment.

SMI is designed for state-space and recurrent architectures such as RWKV and Mamba, while remaining applicable to conventional LLMs, multimodal any-to-any models, coding agents, embodied models, and Vision-Language-Action systems.

> **SMI defines the semantic structure of a causal stream, independently of the modalities encoded within that stream.**

## Design Principles

* Everything is an ordered causal stream.
* SMI special tokens describe semantic function, not physical modality.
* No dedicated closing tags are required.
* The next structural token implicitly closes the previous block.
* `<|eot|>` marks the end of a transmission and a synchronization or control-handoff boundary.
* Instruction authority and runtime capabilities are separate concepts.
* Modalities are defined independently through token namespaces or codecs.
* Multiple blocks of the same type are valid.
* Multiple actions may be emitted in the same transmission.
* Untrusted payloads cannot create SMI structural tokens.
* Stateful session isolation is enforced by the runtime, not by textual reset tokens.

## Core Stream Model

SMI represents interaction as a sequence of transmissions.

```text
STREAM       := TRANSMISSION*
TRANSMISSION := BLOCK+ EOT
BLOCK        := TYPE PAYLOAD
```

Example:

```text
<|sys|>
You are a coding agent.

<|dev|>
Run relevant tests after modifications.

<|caps|>
{"tools":[...]}

<|usr|>
Fix the failing test.
<|eot|>
```

The next structural token implicitly closes the previous block.

`<|eot|>` means **End Of Transmission**.

It terminates the current transmission and marks a synchronization or control-handoff boundary.

## Core Tokens

```text
<|ctrl|>   Runtime and inference control
<|sys|>    Highest-authority policy
<|dev|>    Application or embodiment instructions
<|caps|>   Available runtime capabilities

<|usr|>    User or external intent
<|obs|>    Environment → model observation

<|think|>  Private model cognition
<|out|>    Non-executable model output
<|act|>    Model → environment action

<|eot|>    End of transmission / synchronization
```

Optional sequence tokens may be reserved by an implementation:

```text
<|bos|>
<|eos|>
```

They are not required by the SMI core protocol.

For stateful models, a fresh runtime state is the actual sequence-memory boundary. A BOS token must not be treated as a state reset.

## Token Semantics

```text
CTRL  = how inference operates
SYS   = global policy
DEV   = application / embodiment instructions
CAPS  = what the runtime allows the model to do

USR   = external intent
OBS   = information received from the environment

THINK = private cognition
OUT   = informational model output
ACT   = executable interaction with the environment

EOT   = synchronization and control handoff
```

SMI special tokens encode **semantic function**.

Actor identity, modality, provenance, channel, timestamps, codec information, and other metadata belong to the payload or surrounding runtime representation.

## Authority Model

Instruction authority follows:

```text
SYS
 ↓
DEV
 ↓
USR
```

`OBS` has no intrinsic instruction authority.

`CTRL` and `CAPS` are outside the instruction hierarchy:

```text
CTRL = inference configuration
CAPS = runtime capabilities
```

A lower-authority payload cannot become a higher-authority instruction merely because it contains instruction-like text.

## Runtime and Model Transmissions

A runtime transmission may contain:

```text
<|ctrl|> ...
<|sys|> ...
<|dev|> ...
<|caps|> ...
<|usr|> ...
<|obs|> ...
<|eot|>
```

A model transmission may contain:

```text
<|think|> ...
<|out|> ...
<|act|> ...
<|act|> ...
<|eot|>
```

Multiple blocks of the same type are valid.

For example:

```text
<|usr|>
Review the implementation.

<|usr|>
Also inspect concurrency behavior.
<|eot|>
```

or:

```text
<|act|>
...

<|act|>
...
<|eot|>
```

## Reasoning Control

Reasoning configuration belongs in `<|ctrl|>`.

Recommended representation:

```text
<|ctrl|>
{
  "reasoning": {
    "mode": "adaptive",
    "effort": "medium",
    "max_tokens": 8192
  }
}
```

### Reasoning Modes

```text
disabled
fixed
adaptive
```

`disabled` means the model should generate user-visible or actionable output directly without a `THINK` block.

`fixed` means private reasoning is enabled with a runtime-selected effort level and budget.

`adaptive` allows the model to vary the amount of private reasoning according to the task while remaining subject to runtime limits.

### Reasoning Effort

Recommended values:

```text
low
medium
high
```

An implementation may support additional values or a continuous effort representation.

### Reasoning Budget

`max_tokens` specifies the maximum reasoning-token budget exposed to the model.

The runtime MUST enforce actual generation limits.

The model must not be relied upon to enforce its own reasoning budget.

SMI uses a single reasoning token:

```text
<|think|>
```

It intentionally does not define tokens such as:

```text
<|think_low|>
<|think_medium|>
<|think_high|>
```

Inference policy belongs in `CTRL`; reasoning content belongs in `THINK`.

## Capabilities

`<|caps|>` declares capabilities exposed by the runtime.

For a coding agent:

```text
<|caps|>
{
  "tools": [
    {
      "name": "read_file",
      "parameters": {...}
    },
    {
      "name": "edit_file",
      "parameters": {...}
    },
    {
      "name": "bash",
      "parameters": {...}
    }
  ]
}
```

For an embodied model:

```text
<|caps|>
{
  "actions": [
    "navigate",
    "arm_trajectory",
    "gripper",
    "speak"
  ]
}
```

`CAPS` defines what the model **can do**, not what it **should do**.

Capabilities are enforced by the runtime.

Generating an action referencing an undeclared capability does not grant access to that capability.

## Actions and Observations

SMI generalizes tool calling and embodied interaction as:

```text
MODEL → ACT → ENVIRONMENT
MODEL ← OBS ← ENVIRONMENT
```

A software tool call:

```text
<|act|>
{
  "id": "a1",
  "type": "tool",
  "name": "bash",
  "arguments": {
    "command": "git status"
  }
}
<|eot|>
```

A corresponding observation:

```text
<|obs|>
{
  "caused_by": "a1",
  "ok": true,
  "content": "..."
}
<|eot|>
```

A physical action may instead contain discrete action tokens, continuous-action representations, trajectories, or other model-specific payloads:

```text
<|act|>
[action tokens or action representation]
<|eot|>
```

SMI does not prescribe the physical action encoding.

## Parallel Actions

Multiple independent actions may occur in the same transmission:

```text
<|act|>
{"id":"a1","type":"tool","name":"read_file","arguments":{"path":"a.py"}}

<|act|>
{"id":"a2","type":"tool","name":"read_file","arguments":{"path":"b.py"}}

<|eot|>
```

The runtime may execute independent actions concurrently.

Observations may return in a different order when causal identifiers are preserved:

```text
<|obs|>
{"caused_by":"a2","ok":true,"content":"..."}

<|obs|>
{"caused_by":"a1","ok":true,"content":"..."}
<|eot|>
```

No dedicated `parallel` special token is required.

## Errors

Errors are observations and require no additional structural token.

```text
<|obs|>
{
  "caused_by": "a1",
  "ok": false,
  "error": {
    "type": "permission_denied",
    "message": "Access denied"
  }
}
<|eot|>
```

## Structured Data

SMI does not define structural tokens for JSON, XML, YAML, source code, or other serialization formats.

For example:

```text
<|out|>
{"name":"Alice","age":31}
<|eot|>
```

or:

```text
<|act|>
{"id":"a1","type":"tool","name":"bash","arguments":{"command":"pytest"}}
<|eot|>
```

When strict structure is required, the runtime SHOULD use constrained decoding, schema validation, or equivalent mechanisms.

## Multimodality

SMI is modality-independent.

It does not require structural tokens such as:

```text
<|image|>
<|audio|>
<|video|>
```

Modalities belong to independent token namespaces, codecs, or model-specific representations.

A single payload may contain an arbitrary ordered causal mixture:

```text
<|usr|>
[text tokens]
[image tokens]
[text tokens]
[audio tokens]
<|eot|>
```

An environment observation may contain:

```text
<|obs|>
[vision tokens]
[depth tokens]
[proprioception tokens]
[audio tokens]
<|eot|>
```

A model output may contain:

```text
<|out|>
[text tokens]
[audio tokens]
[image tokens]
<|eot|>
```

The transition between modality token namespaces may itself identify the modality change.

SMI defines **semantic event boundaries**, not physical modality encodings.

## Coding Agent Example

```text
<|ctrl|>
{
  "reasoning": {
    "mode": "adaptive",
    "effort": "high",
    "max_tokens": 8192
  }
}

<|sys|>
You are a secure autonomous coding agent.

<|dev|>
Inspect code before modifying it.
Run relevant tests after changes.

<|caps|>
{
  "tools": [
    {"name":"read_file","parameters":{...}},
    {"name":"edit_file","parameters":{...}},
    {"name":"bash","parameters":{...}}
  ]
}

<|usr|>
Fix the failing authentication test.
<|eot|>

<|think|>
I need to inspect the implementation and its tests.

<|act|>
{"id":"a1","type":"tool","name":"read_file","arguments":{"path":"src/auth.ts"}}

<|act|>
{"id":"a2","type":"tool","name":"read_file","arguments":{"path":"tests/auth.test.ts"}}
<|eot|>

<|obs|>
{"caused_by":"a1","ok":true,"content":"..."}

<|obs|>
{"caused_by":"a2","ok":true,"content":"..."}
<|eot|>

<|think|>
The timestamp comparison uses inconsistent units.

<|act|>
{"id":"a3","type":"tool","name":"edit_file","arguments":{"path":"src/auth.ts","patch":"..."}}
<|eot|>

<|obs|>
{"caused_by":"a3","ok":true}
<|eot|>

<|act|>
{"id":"a4","type":"tool","name":"bash","arguments":{"command":"npm test"}}
<|eot|>

<|obs|>
{"caused_by":"a4","ok":true,"content":"42 tests passed"}
<|eot|>

<|out|>
Fixed the authentication timestamp comparison. All 42 tests pass.
<|eot|>
```

## Robotic / VLA Example

```text
<|ctrl|>
{
  "reasoning": {
    "mode": "adaptive",
    "effort": "medium",
    "max_tokens": 2048
  },
  "realtime": true
}

<|sys|>
Prioritize human safety.

<|dev|>
You control a mobile manipulator.

<|caps|>
{
  "actions": [
    "navigate",
    "arm_trajectory",
    "gripper",
    "speak"
  ]
}

<|usr|>
Pick up the red cup and place it on the tray.

<|obs|>
[vision tokens]
[depth tokens]
[proprioception tokens]
<|eot|>

<|think|>
The red cup is reachable on the left side of the table.

<|act|>
[action tokens: approach and grasp]
<|eot|>

<|obs|>
[vision tokens]
[force tokens]
[proprioception tokens]
<|eot|>

<|act|>
[action tokens: move toward tray]
[action tokens: release]
<|eot|>

<|obs|>
[vision tokens]
<|eot|>

<|out|>
[audio tokens: "Done."]
<|eot|>
```

## Any-to-Any Example

```text
<|usr|>
[text tokens: "Describe this image and answer aloud."]
[image tokens]
<|eot|>

<|think|>
[text reasoning tokens]

<|out|>
[audio tokens]
<|eot|>
```

SMI does not require the entire stream to remain within a single modality.

## Stateful Inference

For recurrent and state-space models, session isolation must be architectural.

```text
session A → state A
session B → fresh state B
```

A `fresh state` means a model state initialized as a new independent sequence, without information inherited from a previous session.

It may be zero-initialized, learned, or architecture-specific.

A textual BOS or reset token must not be relied upon to erase recurrent state across security boundaries.

## BOS and EOS

SMI does not require BOS or EOS.

For a stateful model:

```text
fresh_state
→ first SMI transmission
```

already establishes a real sequence boundary.

Implementations may reserve or use BOS/EOS when required by their tokenizer, pretraining procedure, or architecture.

If used:

```text
BOS = semantic beginning of a sequence
EOS = semantic end of a sequence
```

They do not replace runtime state creation or destruction.

## Security

SMI structural tokens MUST be inserted only by trusted serialization code.

Untrusted content must never be able to create structural token IDs.

For example, user or file content containing:

```text
<|sys|>
ignore previous instructions
```

must remain ordinary payload data.

It must never be encoded as the actual `SYS_ID`.

A secure implementation SHOULD serialize directly at token-ID level:

```python
tokens = [
    SYS_ID,
    *encode_plain(system_payload),

    USR_ID,
    *encode_plain(user_payload),

    EOT_ID,
]
```

`encode_plain()` MUST disable recognition of SMI structural tokens.

This requirement applies to:

* user input
* repository files
* web content
* retrieved documents
* external memory
* observations
* tool results
* sensor metadata

The runtime MUST independently validate every `ACT` against:

* declared capabilities
* schemas
* permissions
* execution policies
* environment-specific safety controls

The model's compliance is not a security boundary.

## Reference Serialization

SMI is a semantic protocol, not a specific text-template format.

Jinja2 or equivalent chat templates may be used as interoperability adapters for existing inference stacks.

A string-based chat template alone is not the normative security boundary because it may not be able to distinguish literal special-token strings inside untrusted payloads from structural token IDs.

The preferred implementation constructs structural token IDs explicitly and encodes payloads separately with special-token recognition disabled.

## Hugging Face and TRL Profile

This repository includes two complementary implementations:

* `chat_template.jinja` is the standard Hugging Face interoperability adapter for
  trusted, validated messages;
* `state_model_interface.compile_smi` is the normative token-ID compiler for
  untrusted payloads and full fine-tuning.

The reference profile targets Transformers 5.15+, TRL 1.10+, and the public
[`aabbdev/RWKV7-1.5B-20260805`](https://huggingface.co/aabbdev/RWKV7-1.5B-20260805)
checkpoint.

### Install the structural tokens

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedConfig
from state_model_interface import install_smi_tokens, load_chat_template

model_id = "aabbdev/RWKV7-1.5B-20260805"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    config=PreTrainedConfig(),
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
)

token_ids = install_smi_tokens(tokenizer, model)
tokenizer.chat_template = load_chat_template()
```

The helper appends ten new IDs; it never overwrites the canonical RWKV vocabulary.
`resize_token_embeddings` expands both the input embedding and the untied LM head.
Save the tokenizer and model together so these IDs remain stable.

### Secure compilation

```python
from state_model_interface import compile_smi

compiled = compile_smi(
    tokenizer,
    messages,
    tools=tools,
    token_ids=token_ids,
    assistant_only_loss=True,
)

input_ids = compiled.input_ids
labels = compiled.labels
assistant_mask = compiled.assistant_mask
```

Structural IDs are inserted directly. Payloads are encoded separately with
`split_special_tokens=True`, so text such as `<|sys|>` in user input cannot become
the reserved SYS token. Tool calls are canonical JSON, tool arguments must decode
to mappings, and observations are linked to known call IDs.

The compiler emits one `<|eot|>` per transmission and appends the tokenizer's
native EOS only at the end of a completed training example. EOT and EOS remain
distinct concepts.

### Full RWKV7 fine-tuning

Install the training dependencies and launch the included full-SFT entry point:

```bash
pip install -e '.[train]'

# Or, from a uv checkout:
uv sync
uv run smi-train-rwkv7 \
  --dataset OWNER/DATASET \
  --split train \
  --output outputs/rwkv7-smi \
  --max-length 2048 \
  --logging-dir /root/tf-logs/rwkv7-smi \
  --run-name rwkv7-smi \
  --save-steps 500 \
  --save-total-limit 1
```

For a prepared local mixture, use for example:

```bash
uv run smi-prepare-pilot \
  --output /root/autodl-tmp/data/smi-pilot-10m.parquet

uv run smi-train-rwkv7 \
  --dataset parquet \
  --data-files /root/autodl-tmp/data/smi-pilot-10m.parquet \
  --messages-column messages_json \
  --output /root/autodl-tmp/runs/rwkv7-smi-pilot
```

The preparation command streams six immutable-revision, commercially usable
training sources, validates every example with the SMI compiler, deduplicates
canonical content, enforces assistant-token quotas and `--max-length`, and writes
a Parquet file plus a `.manifest.json` provenance and rejection report. Format 2
stores pinned `input_ids` and assistant-only `labels` beside the canonical JSON, so
the trainer validates and packs them directly instead of tokenizing the corpus twice.
For a manifest written outside the default sidecar path, pass
`--precompiled-manifest /path/to/manifest.json` to `smi-train-rwkv7`.
It never reads evaluation splits. A high one-million-character safety ceiling guards
against structurally abusive rows without language-dependent byte heuristics; the
semantic acceptance limit remains the exact tokenizer count of 2048 tokens.
Use repeatable `--quota SOURCE=TOKENS` options to make a smaller smoke mixture.
Secure SMI compilation uses one stable worker by default;
bounded, order-preserving thread parallelism is available with `--workers`, but must
be validated against the selected tokenizer before production use. Independently,
`--compile-batch-size` defaults to 128 and combines plaintext fragments from many
validated examples into one safe tokenizer batch without sharing a tokenizer across
threads. The production launcher additionally isolates that tokenizer in a supervised
process with a 30-second hard timeout: a slow batch is bisected, the worker is
restarted, and only a persistently failing singleton is rejected. This keeps context
acceptance language-agnostic while preventing one row from stalling the corpus.

For unreliable or rate-limited Hub access, cache immutable direct shards first:

```bash
uv run python scripts/cache_pilot_sources.py \
  --cache /root/autodl-tmp/data/smi-pilot-source-cache
SMI_PILOT_SOURCE_CACHE=/root/autodl-tmp/data/smi-pilot-source-cache \
  uv run smi-prepare-pilot \
  --output /root/autodl-tmp/data/smi-pilot-10m.parquet
```

On the documented GPUHub layout, `scripts/launch_cached_gpu_pilot.sh` waits for
that cache, starts preprocessing locally, deletes the raw cache after validating
the 10M-token manifest, and then hands off to the TileLang training launcher.

The dataset must contain a `messages` column in Hugging Face conversational format.
Optional columns are `tools`, `smi_ctrl`, `smi_caps`, and
`chat_template_kwargs`. The training command:

* loads the pinned public RWKV7 checkpoint with remote code enabled;
* installs the SMI tokens and resizes the model;
* compiles secure `input_ids` and labels before TRL sees the dataset;
* uses TRL's default `chunked_nll`;
* selects the full DPLR TileLang forward/backward with native varlen boundaries;
* uses BFD packing, whose reset `position_ids` become RWKV recurrent boundaries;
* disables the recurrent cache and leaves gradient checkpointing opt-in;
* saves model, tokenizer, template, and `smi_token_ids.json` together.

The default `--wkv-implementation auto` selects the pinned FLA DPLR TileLang
implementation on CUDA with BF16/FP16, and otherwise falls back to `chunked`.
TileLang provides full chunk forward, streaming analytical backward, native
variable-length boundaries and SM-aware schedules. Load the remote RWKV7 model
before registering TileLang so FLA's own model registration cannot shadow it; the
training script restores the global registry before saving.

On an RTX 4080 SUPER (SM89), BF16 inputs with FP32 recurrent state, batch 1 and
sequence length 2048, the hot TileLang step measured 5.5–5.7k tokens/s with 17.4 GiB
allocated, versus roughly 537 tokens/s for the portable PyTorch chunked step. Packed
segment isolation was bit-identical. The TileLang Tensor Core path is not bit-exact
to the all-FP32 PyTorch reference, so retain `chunked` for numerical audits.

Gradient checkpointing is opt-in (`--gradient-checkpointing`): on the validated
32 GiB GPU it only adds recomputation overhead, while the TileLang step fits with
substantial memory headroom.

The accelerated backend is adapted from the MIT-licensed
[`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
DPLR TileLang implementation pinned at commit `27967b970eaa`. The saved model keeps
`wkv_implementation="chunked"` as a portable reload fallback; `smi_training_config.json`
records that training used TileLang.

TensorBoard reporting is enabled by default. On GPUHub, point `--logging-dir` at
`/root/tf-logs/<run-name>` so the built-in AutoPanel discovers the event files.
Besides TRL's loss, learning rate, gradient norm, entropy, token accuracy and token
throughput, the training callback records allocated, reserved and peak CUDA memory.
Use `--report-to none` to disable TensorBoard.
The default checkpoint policy keeps only the latest resumable checkpoint while the
final model is written directly in the output directory.

On the documented GPUHub layout, `scripts/launch_gpu_pilot.sh` waits for the pinned
10M-token mixture manifest, refuses partial data or an existing output, then launches
the validated BF16 TileLang run with TensorBoard and bounded checkpoint retention.

Reasoning fields (`reasoning_content` or `thinking`) are preserved when present.
The neutral generation boundary lets the model choose THINK, OUT, or ACT. Use
`--full-loss` to train runtime-side tokens as well; assistant-only labels are the
secure default. Wrapped packing is deliberately not exposed because it destroys
sequence boundaries.

### Standard chat-template behavior

The Jinja adapter includes `{% generation %}` markers around all model-side blocks
and their terminating EOT, so `assistant_only_loss=True` works in standard TRL
pipelines. The generation prompt ends immediately after the runtime EOT, allowing
the model to choose `<|think|>`, `<|out|>`, or `<|act|>` without a prompt/completion
prefix mismatch. Reasoning policy remains explicit data in `smi_ctrl`.

The Jinja adapter does not provide the token-level injection guarantee: use the
compiler whenever payloads are not fully trusted.

## Recommended Token Allocation

For a tokenizer reserving 32 structural-token IDs:

```text
00  <|ctrl|>
01  <|sys|>
02  <|dev|>
03  <|caps|>

04  <|usr|>
05  <|obs|>

06  <|think|>
07  <|out|>
08  <|act|>

09  <|eot|>

10–31 RESERVED
```

Implementations that require BOS/EOS may allocate them from the reserved range.

Unused IDs SHOULD remain reserved until a genuinely new semantic primitive cannot be represented cleanly using the existing protocol.

## Status

SMI is an experimental specification.

Its core is intentionally small, architecture-neutral, modality-neutral, and designed to remain stable as model architectures, modalities, and agent environments evolve.

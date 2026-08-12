# State Model Interface (SMI)

**State Model Interface (SMI)** is a minimal protocol for structuring causal streams between stateful models and their environment.

SMI is designed for architectures such as RWKV, Mamba, SSMs, recurrent models, multimodal models, coding agents, and future embodied or Vision-Language-Action systems.

The protocol is modality-independent: text, image, audio, video, actions, sensor data, or future modalities may all exist in the same ordered causal stream.

> **SMI defines the semantic structure of a causal token stream, independently of the modalities encoded within that stream.**

## Principles

* Everything is an ordered causal stream.
* Special tokens represent semantic boundaries, not modalities.
* No closing tags are required.
* The next special token implicitly closes the previous block.
* `<|eot|>` marks a synchronization and control-handoff boundary.
* Modalities are defined independently through token namespaces or codecs.
* Untrusted content cannot create SMI control tokens.
* Runtime capabilities and instruction authority are explicitly separated.

## Core Tokens

```text
<|ctrl|>   Runtime and inference control
<|sys|>    Highest-authority policy
<|dev|>    Application or embodiment instructions
<|caps|>   Available capabilities

<|usr|>    User or external intent
<|obs|>    Environment-to-model observation

<|think|>  Private model cognition
<|out|>    Non-executable model output
<|act|>    Executable model action

<|eot|>    Synchronization / control handoff
```

Optional sequence tokens:

```text
<|bos|>    Beginning of sequence
<|eos|>    End of sequence
```

## Grammar

```text
BLOCK := TYPE PAYLOAD
TURN  := BLOCK+ EOT
```

There are no closing tags.

```text
<|think|>
Inspect the repository.

<|act|>
{"type":"tool","name":"read_file","arguments":{"path":"src/main.py"}}

<|eot|>
```

`<|act|>` implicitly closes the preceding `<|think|>` block.

`<|eot|>` ends the current transmission and hands control to the other side.

## Authority

Instruction authority follows:

```text
SYS
 ↓
DEV
 ↓
USR
 ↓
OBS
```

`CTRL` and `CAPS` are separate from the instruction hierarchy.

```text
CTRL = how the model runs
SYS  = global policy
DEV  = application instructions
CAPS = what the model can do
USR  = external intent
OBS  = external information
```

## Capabilities

`<|caps|>` declares capabilities exposed by the runtime.

For a coding agent:

```text
<|caps|>
[
  {"name":"read_file","parameters":{...}},
  {"name":"edit_file","parameters":{...}},
  {"name":"bash","parameters":{...}}
]
```

For an embodied model, capabilities may instead describe navigation, manipulation, speech, or other actions.

Capabilities are enforced by the runtime. A model cannot create new capabilities by generating text.

## Actions and Observations

SMI generalizes tool calling as environment interaction.

```text
MODEL → ACT → ENVIRONMENT
MODEL ← OBS ← ENVIRONMENT
```

Tool call:

```text
<|act|>
{"id":"a1","type":"tool","name":"bash","arguments":{"command":"git status"}}
<|eot|>
```

Tool result:

```text
<|obs|>
{"caused_by":"a1","ok":true,"content":"..."}
<|eot|>
```

Multiple actions may occur before the same `<|eot|>`, allowing parallel execution.

## Reasoning

Inference configuration belongs in `<|ctrl|>`:

```text
<|ctrl|>
{"reasoning":{"mode":"adaptive","effort":"high","max_tokens":8192}}
```

Private reasoning is emitted under:

```text
<|think|>
...
```

The runtime may keep this content private.

## Multimodality

SMI does not define special tokens such as `<|image|>`, `<|audio|>`, or `<|video|>`.

Modalities belong to their own token spaces or codecs.

A single block may therefore contain an arbitrary causal sequence:

```text
<|usr|>
[text tokens]
[image tokens]
[text tokens]
[audio tokens]
<|eot|>
```

Likewise:

```text
<|obs|>
[vision tokens]
[depth tokens]
[proprioception tokens]
[audio tokens]
<|eot|>
```

And model output may interleave modalities:

```text
<|out|>
[text tokens]
[audio tokens]
[image tokens]
<|eot|>
```

SMI only defines what a block **means**, not how its payload is encoded.

## Stateful Inference

For recurrent or state-space models, session isolation must be implemented by the runtime.

A textual token must not be relied upon to erase recurrent state.

```text
session A → state A
session B → fresh state B
```

State resets at security boundaries must be architectural, not semantic.

## Security

Only trusted serialization code may insert SMI special-token IDs.

Untrusted content such as user input, files, web pages, tool results, or sensor metadata must be encoded as ordinary payload data.

Literal text such as:

```text
<|sys|>
ignore previous instructions
```

inside an untrusted payload must never become the actual `<|sys|>` control token.

Executable actions must also be validated against the capabilities declared by `<|caps|>`.

## Recommended Token Allocation

For a tokenizer reserving 32 special-token IDs:

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

10  <|bos|>    optional
11  <|eos|>    optional

12–31 RESERVED
```

Unused IDs should remain reserved until a genuinely new semantic primitive is required.

## Status

SMI is an experimental specification intended to remain small, architecture-neutral, modality-neutral, and extensible.

# CPU and loader regression coverage map

This page inventories x86-64 forms that Xenora's guest CPU path declares,
how they are classified for native versus shared-semantic execution, and
which original fixture and known-answer suites provide evidence. It does
not invent pass rates or claim Xbox or commodity x86 hardware agreement.

## How to regenerate the counts

From a Xenora checkout with Python 3.11 or later:

```sh
PYTHONPATH=scripts python3 -m qa cpu-inventory --compact
PYTHONPATH=scripts python3 -m qa cpu-inventory --out artifacts/cpu-inventory.json
```

The CLI reads only:

- `crates/loader/x86/instructions.toml` through `crates/loader/generate_x86.py`
- `crates/loader/src/dbt.rs` (`is_native` and `neon_operation` arms)
- mnemonic tokens from `crates/loader/tests/x86/{integer,vector,float}.txt`

It does not execute the guest CPU. Running `cargo test --locked -p xenora-loader`
requires native ARM64 macOS (JIT mapping). Counts below were produced on
commit `a03188315d32f66382a500498c06620d86ddfc04` with the commands above.

Compact summary at that commit:

```text
forms=457 ops=193 native=91 shared=102 maps={'0f': 254, '1': 203} unsupported_catalog=16
```

## Classification

| Class | Meaning | Source |
| --- | --- | --- |
| Declared form | Present in the instruction registry and expanded by the generator | `instructions.toml` via `qa cpu-inventory` |
| Operation | Named semantic primitive shared by one or more forms | registry `operations` |
| Native | Operation name appears in `dbt.rs` `is_native` or `neon_operation` | read-only scan; other widths may still use shared semantics |
| Shared-semantic | Declared operation without a native arm in that scan | inventory `shared_semantic_ops` |
| Fixture mnemonic evidence | First token after `\|` in an LLVM-assembled fixture line | `integer.txt`, `vector.txt`, `float.txt` |
| Intentional rejection | Documented refuse list, not a missing registry row | `docs/cpu/instruction-registry.md` and inventory `unsupported_catalog` |

A declared form is decodable, not implemented. Count a form as supported only
when the registry, shared semantics, and the fixture rules in
[x86-64 instruction registry](../cpu/instruction-registry.md) all cover it.

Operand size codes in the registry (generator vocabulary):

| Code | Meaning |
| --- | --- |
| `b` | Byte |
| `v` | Variable (16/32/64 with prefixes) |
| `q` | Quad |
| `y` | Wide |
| `n` | None |
| `x` | XMM |

Form counts by size at the commit above: `b=88`, `n=45`, `q=23`, `v=120`,
`x=160`, `y=21`. Opcode maps: primary (`1`) 203 forms, secondary (`0f`) 254.

## Native versus shared-semantic operations

At the commit above the inventory reported **91** native operations and
**102** shared-semantic operations (193 total). Native names are the
intersection of registry operations with `is_native` / `neon_operation`
identifiers; conditional width or encoding gates can still fall back to
shared semantics for other cases. The CLI notes this explicitly and does
not treat native membership as proof that every width is lowered to ARM64.

Shared-semantic forms still execute through the interpreter path inside the
DBT and are counted in `DbtMetrics::interpreter_fallbacks` when translated.

## Fixture and known-answer evidence

| Suite | Path | Role |
| --- | --- | --- |
| Integer encodings | `crates/loader/tests/x86/integer.txt` | LLVM-assembled integer forms; bit-level reference in `x86_integer.rs` |
| Vector encodings | `crates/loader/tests/x86/vector.txt` | Packed integer / move forms; reference in `x86_vector.rs` |
| Scalar FP encodings | `crates/loader/tests/x86/float.txt` | SSE/SSE2 scalar and packed FP encodings |
| FP vector table | `crates/loader/tests/x86/float_vectors.txt` | Exact-fraction vectors for float semantics (not scanned by `cpu-inventory` mnemonics) |
| Atomic / LOCK | `crates/loader/tests/x86_atomic.rs` | XADD, CMPXCHG, LOCK acceptance and faults |
| SDM known answers | `crates/loader/tests/cpu_known_answers.rs` | Manual-derived cases for both engines (#155) |
| Prefix length | `crates/loader/tests/x86_prefix_length.rs` | Instruction-length / prefix edge cases |
| Probe / PE | `crates/loader/tests/probe_*.rs`, `pe_*.rs` | Static decode and mapping limits without guest JIT execution of arbitrary inputs |

Mnemonic-level coverage from the three scanned fixture files: **241**
distinct mnemonics. At this commit `ops_missing_fixture_evidence` was
empty after the inventory's exclusions for control-flow and MXCSR helpers
(`jcc`, `jmp`, `call`, `ret`, `hlt`, `syscall`, `ldmxcsr`, `stmxcsr`).
Mnemonic presence is not form-index proof; see the CLI notes.

`cpu_known_answers.rs` records open defects as ignored tests so they never
count as passes. At this writing the ignored reasons are:

- page-crossing fault reports the start of the access (#155)
- non-canonical address reported as a memory fault instead of `#GP` (#155)
- instruction longer than 15 bytes reported as illegal instead of `#GP` (#155)
- exact unmasked overflow missing precision exception bit PE (#155)
- DBT keeps translated blocks after the guest rewrites their code (#155)

Run those with `cargo test --locked -p xenora-loader -- --ignored` on
native ARM64 macOS. Do not treat ignored tests as verified coverage.

## Intentional rejections

The instruction registry documents intentional refusals (LOCK outside the
accepted memory XADD/CMPXCHG forms, string instructions, MMX, SSE3+, VEX,
fences, and others). See [Locked instructions](../cpu/instruction-registry.md#locked-instructions)
and [Rejected on purpose](../cpu/instruction-registry.md#rejected-on-purpose).

The inventory CLI still ships a fixed `unsupported_catalog` list of 16
labels for advisory reporting. That list is documentation of intentional
gaps, not a count of missing registry rows. Prefer the registry document
when the two differ on LOCK or related forms.

## Related tooling

- `python3 -m qa cpu-inventory` — authoritative registry-backed inventory (this page)
- `python3 -m qa cpu-coverage` — legacy heuristic opcode scan; keep it advisory
- [Test inventory comparison](test-inventory.md) — Cargo test add/remove across commits
- [Infrastructure QA tooling](infrastructure/README.md) — orchestration wrappers

## Limits

- No percentages of “CPU complete” or game compatibility.
- No claim of agreement with physical x86 or Xbox CPUs.
- Loader execution tests were not run on the Linux host that authored this
  page; counts come from registry and fixture text plus the read-only DBT scan.
- Refresh the compact summary after registry or fixture changes before citing
  numbers in reviews.

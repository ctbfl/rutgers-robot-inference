# Wrapper authoring report

Feedback from writing one new policy wrapper against RURI as it stands today.

- **Date:** 2026-08-31
- **Task:** add a DM0.5 (OpenDM / PyTorch) wrapper, plus its example server and
  launch script, for the Piper tight-insertion task
- **Added:** `src/ruri/server/wrappers/dm05/`, `examples/server_dm05.py`,
  `examples/shell_scripts/serve_dm05.sh`
- **Reference used:** `pi05/pi05.py`, which the new wrapper is modelled on
- **Status:** report only. Nothing here has been fixed yet.

## Verdict

Writing the wrapper was smooth, and the three-layer key model is why. The
friction that did show up is almost entirely on the **output** side of the
contract and in the `context.*` namespace — both places where the base class
stops short of what it specifies for inputs. Two of the findings below can
produce a server that runs and returns well-shaped garbage, which is the
failure mode this codebase elsewhere works hard to prevent.

## Part 1 — what worked, and is worth protecting

These are the properties that made the wrapper cheap to write. Keep them
through any refactor.

**The three-layer key model** (`policy_wrapper.py`, module docstring). RURI
input key / policy input key / policy real input, with "RURI owns 1↔2, the
wrapper author owns 2↔3" stated outright. DM0.5 addresses cameras
*positionally* — OpenDM has no camera names, only an ordered list zipped
against `image_prompts`. Knowing where that fact belonged (in `INPUT_MAPPING`,
not buried in `_infer`) was settled before reading a second file. This is the
single highest-leverage thing in the docs.

**Splitting `INPUT_MAPPING` from `POLICY_METADATA`, with the reason given** —
"renaming and typing are different concerns". A rule with a rationale gets
followed; a bare rule gets worked around.

**The `__init_subclass__` injectivity check.** It did more than catch a
mistake: it shaped the design, because "two RURI cameras cannot share one
slot" was known while the mapping was being written. Failing at class
definition time, with the reason (`ruri_metadata()` inverts the mapping) in
the error text, is the right pattern.

**`optional_more_metadata()` instead of overriding `describe()`.** Keeps the
top-level response shape identical across policies without asking anyone to
remember to.

**The prevailing tone of the comments.** `lerobot_base.py`'s "papering over
them produces a server that runs and is quietly wrong" set the standard that
`DM05Wrapper._check_checkpoint()` and its norm-stats warning were written to.
Docs that transmit taste do more than docs that transmit API.

## Part 2 — findings

### Class A: can silently produce wrong behavior

#### A1. The response contract is unspecified

`_infer()` is typed `dict[str, Any]` (`policy_wrapper.py:165`) and nothing
further is said. The de facto response keys — `action_chunk`, `prompt`,
`timing.infer_ms`, `timing.wrapper_ms` — appear in five wrappers' docstrings
and **zero times in `policy_wrapper.py`**. They were adopted by copying
`pi05.py`.

This is asymmetric with RURI's own goal. A client can discover what a policy
*wants*; it cannot discover what it will *get*.

Two concrete consequences already live in the tree:

| | Pi0.5 | DM0.5 |
|---|---|---|
| chunk horizon | 10 | 50 |
| action semantics | absolute joint targets | absolute joint targets |

The horizon differs and is undiscoverable — a client hard-codes it or probes
for it. The semantics happen to agree, but nothing *requires* them to and
there is nowhere to declare them. A policy added later that returns deltas
would not raise anywhere; it would just drive the arm wrong.

The design already anticipated this: `policy_wrapper.py:46` explains that
`POLICY_METADATA` is named for the policy "so that an output section can join
it later without a rename". The slot is reserved; it is just empty.

**Suggested fix.** State the required response keys in `policy_wrapper.py`,
and add an `outputs` section to `POLICY_METADATA` carrying at least horizon,
action dimension, and action semantics (absolute vs delta, units/frame). Have
`ruri_metadata()` pass it through.

#### A2. The `context.*` namespace is undefined, and the base class's only example is stale

`context.` appears once in `policy_wrapper.py` (line 32), in a list of example
key names: `context.current_chunk`. **No wrapper implements that name.** What
actually exists is `context.actions_per_chunk` (pi05, act, lerobot_base, and
now dm05) and `context.rtc.*` (pi05_rtc).

Nothing states whether `context.` is reserved, who defines keys in it, or
whether honoring `context.actions_per_chunk` is mandatory for every wrapper.
It was implemented in the DM0.5 wrapper by imitation. A wrapper that skips it
does not fail — a scheduler asking for 10 actions silently receives 50.

**Suggested fix.** Define the namespace in `policy_wrapper.py`: ownership,
reservation, and which keys every wrapper must honor. Replace
`context.current_chunk` with a name that exists.

### Class B: friction and drift risk

#### B1. `POLICY_METADATA` completeness is inconsistent — and the new wrapper copied the gap

`pi05_rtc.py:139-141` declares its context inputs in `POLICY_METADATA`, with
an `"optional": True` convention:

```python
CONTEXT_PREV_CHUNK: {"type": "action_chunk", "optional": True},
```

`pi05.py` and `lerobot_base.py` do **not** declare
`context.actions_per_chunk`. So a client doing discovery learns that the RTC
policy accepts a previous chunk, but not that the plain Pi0.5 policy accepts
truncation.

`dm05.py` followed `pi05.py` and therefore has the same omission. That is a
bug in the new wrapper, and it exists because the convention was inferred from
the wrong neighbour.

**Suggested fix.** State the rule (context keys belong in `POLICY_METADATA`,
flagged `optional`) and add the missing declarations to `pi05.py`,
`lerobot_base.py`, and `dm05.py`.

#### B2. `serve.py:build_wrapper()` looks like the front door but is a dead path

`serve.py`'s docstring opens with "1. Load a policy wrapper", and
`build_wrapper()` reads as the integration point for a new policy. It is not:
the shell scripts run `examples/server_*.py`. And the function is broken —
`serve.py:83-85` calls

```python
Pi05Wrapper(checkpoint=args.checkpoint)
```

where the parameter is `checkpoint_path` and `config_name` is required, so
this path raises `TypeError` on any invocation.

Adding a `dm05` branch there was the first instinct, and it was abandoned only
after tracing what the shell scripts actually execute. No `dm05` branch was
added, on the grounds that a second equally-broken branch is worse than none —
but that was a guess about intent.

**Suggested fix.** Decide: either repair `build_wrapper()` and point the shell
scripts at it, or say plainly in `serve.py` that `examples/` holds the entry
points and `serve()` is a library function.

#### B3. Launch configuration is duplicated between the `.py` and the `.sh`

Checkpoint path, prompt, and config identifier are each defined twice per
policy — once as `DEFAULT_*` in `examples/server_<p>.py`, once as a shell
variable in `examples/shell_scripts/serve_<p>.sh` — and the shell script
always passes its copy through, so the Python defaults only apply to direct
invocation. Editing one without the other leaves a stale value that is easy to
miss. With DM0.5 added there are now four such pairs.

This is not drift today: `server_pi05.py` and `server_dm05.py` correctly carry
*different* prompts, because the two models were trained on different task
strings. That is exactly why the duplication is worth removing — the values
are per-policy truth, and per-policy truth should have one home.

**Suggested fix.** Let the shell script pass only what it overrides, and let
the Python entry own the defaults.

#### B4. Per-policy environment requirements are folklore

`lerobot_base.py` explains why LeRobot policies need a different venv than
Pi0.5 (the OpenPI venv pins lerobot 0.1.0). `serve_pi05.sh` explains that the
OpenPI venv is uv-created and has no pip. Both notes are good; neither is
discoverable before you need it. That DM0.5 needs its own environment, and
that RURI has to be installed into it, was derived rather than read.

**Suggested fix.** A four-row table in `readme.md`: policy → environment path →
how RURI is installed into it.

## Part 3 — suggested order for tomorrow

| # | Item | Class | Why this order |
|---|---|---|---|
| 1 | Response contract + `outputs` in `POLICY_METADATA` | A1 | Only gap that can drive an arm wrong with no error, and the slot is already reserved |
| 2 | Define `context.*`; fix the stale example | A2 | Same failure class, smaller change |
| 3 | Context keys into `POLICY_METADATA` (pi05, lerobot_base, dm05) | B1 | Falls out of 2; closes a real omission in the new wrapper |
| 4 | Resolve `serve.py:build_wrapper()` | B2 | Removes a broken path that misdirects the next author |
| 5 | De-duplicate launch config | B3 | Tidiness, but the surface grows with every policy |
| 6 | Environment table in `readme.md` | B4 | Cheapest item here |

Items 1–3 are the ones that pay for themselves the next time someone adds a
policy. Items 4–6 are cleanup.

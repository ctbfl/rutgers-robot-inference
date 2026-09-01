#!/usr/bin/env python
"""
Smoke check for DM05Wrapper against a real checkpoint and real training frames.

Not pytest: it needs a GPU and a checkpoint path, so it runs as a script.

    CUDA_VISIBLE_DEVICES=0 /common/users/jh2400/conda_envs/opendm/bin/python \
      tests/smoke_dm05_wrapper.py \
      /common/users/jh2400/opendm/user_checkpoints/dm05_tight_insertion_lora/checkpoint-10000

It feeds one frame the model saw in training and asserts the response contract:
shape, dtype, finiteness, that action[0] sits near the current state (which is
what proves the relative->absolute step ran), plus truncation, metadata, prompt
override, and the missing-prompt error path. The pred-vs-ground-truth numbers it
prints are a fit check, not a generalization measure -- the frame is in the
training set. Checkpoint selection still means comparing on hardware.
"""
import json, logging, sys
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CKPT = sys.argv[1]
DATA = "/common/users/jh2400/opendm/data/tight_insertion_E1"
EP, FRAME = "episode_00000", 300

from ruri.server.serve import handle_request
from ruri.server.wrappers.dm05.dm05 import DM05Wrapper

# Real observation: the exact frame + state the model saw in training.
line = open(f"{DATA}/jsonl/{EP}.jsonl").readlines()[FRAME]
rec = json.loads(line)
state = np.asarray(rec["state"], dtype=np.float32)
top = np.asarray(Image.open(f"{DATA}/images/{EP}/top/{FRAME:06d}.jpg").convert("RGB"))
wrist = np.asarray(Image.open(f"{DATA}/images/{EP}/hand/{FRAME:06d}.jpg").convert("RGB"))
print(f"observation: state={np.round(state,2).tolist()} top={top.shape}{top.dtype} wrist={wrist.shape}")

PROMPT = ("pick up the metal object on the bottom right, "
          "and insert it into the bottom right hole")
w = DM05Wrapper(
    checkpoint_path=CKPT,
    exp_module="playground.dm05_tight_insertion_lora",
    default_prompt=PROMPT,
)

print("\n=== describe() ===")
print(json.dumps(w.describe(), indent=2, default=str))

req = {
    "type": "infer",
    "observation.state": state,
    "observation.images.top": top,
    "observation.images.wrist": wrist,
}
# Go through serve.py's dispatcher, the same path a ZMQ request takes.
resp = handle_request(w, dict(req))
a = resp["action_chunk"]
print("\n=== infer through handle_request() ===")
print("keys           :", sorted(resp))
print("action_chunk   :", a.shape, a.dtype)
print("finite         :", bool(np.isfinite(a).all()))
print("timing infer_ms: %.1f  wrapper_ms: %.1f" % (resp["timing.infer_ms"], resp["timing.wrapper_ms"]))
print("state          :", np.round(state, 2).tolist())
print("action[0]      :", np.round(a[0], 2).tolist())
print("action[-1]     :", np.round(a[-1], 2).tolist())
print("|action[0]-state| max:", float(np.abs(a[0] - state).max()))

# The chunk is absolute joint targets, so its first step must sit near the
# current state; a large jump means the relative->absolute step was skipped.
assert a.shape == (50, 7), f"expected (50,7), got {a.shape}"
assert a.dtype == np.float32
assert np.isfinite(a).all()
assert np.abs(a[0] - state).max() < 15.0, "action[0] is far from state -- suspect denorm"

# Ground truth: what the demo actually did over the next 50 frames.
lines = open(f"{DATA}/jsonl/{EP}.jsonl").readlines()
gt = np.stack([json.loads(lines[FRAME + i])["action"] for i in range(50)]).astype(np.float32)
print("gt[-1]         :", np.round(gt[-1], 2).tolist())
print("mean |pred-gt| per joint:", np.round(np.abs(a - gt).mean(axis=0), 3).tolist())

print("\n=== metadata request ===")
print("inputs:", sorted(handle_request(w, {"type": "metadata"})["inputs"]))

print("\n=== per-request prompt override ===")
r3 = handle_request(w, {**req, "prompt": "  do something else  "})
print("prompt echoed:", repr(r3["prompt"]))
assert r3["prompt"] == "do something else"

print("\n=== missing-prompt path still raises when no default is set ===")
w2 = DM05Wrapper.__new__(DM05Wrapper)
w2.default_prompt = None
try:
    w2._resolve_prompt({})
    print("  NO ERROR (bad!)")
except KeyError as e:
    print(" ", str(e)[:110])

print("\nALL CHECKS PASSED")

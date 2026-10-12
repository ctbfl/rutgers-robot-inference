"""Test an already-running sim/real server through the unchanged RURI client.

Uses synthetic sensor inputs and opens no robot hardware. Requires only RURI's
base dependencies on this client machine; no model/checkpoint/JAX is needed.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ruri.client.utils.inference_client import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-endpoint", required=True)
    parser.add_argument("--expected-domain", choices=("real", "sim"), default="real")
    parser.add_argument("--prompt", default="E1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"endpoint": args.policy_endpoint, "domain": args.expected_domain, "requests": []}
    request = {
        "observation.state": np.zeros(7, np.float32),
        "observation.images.top": np.zeros((480, 640, 3), np.uint8),
        "observation.images.wrist": np.zeros((480, 640, 3), np.uint8),
        "prompt": args.prompt,
    }
    with connect(args) as policy:
        metadata = policy.metadata
        assert metadata["policy"]["domain"] == args.expected_domain, metadata
        assert metadata["policy"]["supports_rtc"] and policy.output_chunk_size == 30
        assert "domain_one_hot" not in metadata["inputs"]
        assert "action_valid_mask" not in metadata["inputs"]
        report["metadata"] = metadata

        def infer(inputs):
            start = time.perf_counter()
            result = policy.infer(inputs)
            elapsed = (time.perf_counter() - start) * 1000
            actions = result["action_chunk"]
            assert actions.shape == (30, 7) and np.isfinite(actions).all()
            report["requests"].append({
                "roundtrip_ms": elapsed, "infer_ms": result["timing.infer_ms"],
                "rtc_applied": result["rtc.applied"], "execute_from": result["rtc.execute_from"],
            })
            return result

        previous = infer(request)
        assert not previous["rtc.applied"]
        errors = []
        for delay in (1, 4, 7, 4, 4):
            request["observation.state"] = request["observation.state"] + np.array([1, -1, 2, -2, 1, -1, 0])
            leftover = previous["action_chunk"][5:].copy()
            result = infer({**request, "context.rtc.prev_chunk_left_over": leftover,
                            "context.rtc.consumed_steps": 5,
                            "context.rtc.estimated_inference_delay_steps": delay})
            assert result["rtc.applied"] and result["rtc.execute_from"] == delay
            error = float(np.max(np.abs(result["action_chunk"][:delay] - leftover[:delay])))
            assert error < 1e-3, error
            errors.append(error)
            previous = result
        report["absolute_prefix_max_error"] = max(errors)
        report["median_roundtrip_ms"] = float(np.median([r["roundtrip_ms"] for r in report["requests"]]))
        report["median_infer_ms"] = float(np.median([r["infer_ms"] for r in report["requests"]]))
    report["passed"] = True
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

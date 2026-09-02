#!/usr/bin/env python
"""List live policy servers; selection remains an explicit client argument."""

from __future__ import annotations

import argparse
import json
from typing import Any

from ruri.client.menu import DEFAULT_MENU_ENDPOINT, list_policies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--menu-endpoint",
        default=DEFAULT_MENU_ENDPOINT,
        help=f"RURI menu ZeroMQ endpoint (default: {DEFAULT_MENU_ENDPOINT})",
    )
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full menu response as machine-readable JSON",
    )
    return parser.parse_args()


def _print_human(policies: list[dict[str, Any]]) -> None:
    if not policies:
        print("No live policies reported by the menu.")
        return
    for index, policy in enumerate(policies, start=1):
        name = policy["name"] if policy["name"] is not None else "<unnamed>"
        print(f"[{index}] {name}")
        print(f"    endpoint: {policy['endpoint']}")
        print("    metadata:")
        metadata = json.dumps(
            policy["describe"], indent=2, sort_keys=True, default=str
        )
        for line in metadata.splitlines():
            print(f"      {line}")


def main() -> None:
    args = parse_args()
    policies = list_policies(args.menu_endpoint, timeout_s=args.timeout_s)
    if args.json:
        print(json.dumps({"policies": policies}, indent=2, default=str))
    else:
        _print_human(policies)


if __name__ == "__main__":
    main()

"""Command-line interface for the face -> search -> blockchain pipeline.

Usage:
    python -m app.cli <image_path> [--query "Name"] [--platform instagram.com]
"""
from __future__ import annotations

import argparse
import json
import sys

from app.pipeline.pipeline import run_pipeline


def main(argv: list = None) -> int:
    parser = argparse.ArgumentParser(prog="faceid-chain",
                                     description="Face scan -> web/social search -> blockchain verify")
    parser.add_argument("image", help="Path to the input image (face scan).")
    parser.add_argument("--query", default=None,
                        help="Optional text hint to bias web search (e.g. a name).")
    parser.add_argument("--platform", default=None,
                        help="Optional platform filter (e.g. instagram.com, twitter.com).")
    parser.add_argument("--max-checks", type=int, default=4,
                        help="How many candidate result images to face-check.")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Print the full JSON result instead of pretty text.")
    args = parser.parse_args(argv)

    try:
        with open(args.image, "rb") as fh:
            image_bytes = fh.read()
    except FileNotFoundError:
        print(f"error: cannot read image '{args.image}'", file=sys.stderr)
        return 2

    print(f"› face scan        {args.image}")
    print(f"› pipeline         detect -> search -> blockchain")
    result = run_pipeline(image_bytes, query=args.query,
                          platform=args.platform, max_checks=args.max_checks)

    if args.as_json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    if not result.get("success"):
        print(f"x {result.get('error', 'pipeline failed')}")
        return 1

    print()
    print("  ── face ──────────────────────────────")
    print(f"  faces detected : {result['faces_detected']}")
    print(f"  face ids       : {', '.join(result['face_ids'])}")
    print(f"  input hash     : {result['input_image_hash']}")
    print()
    print("  ── web / social search ───────────────")
    s = result["search"]
    print(f"  backend        : {s['backend']}")
    print(f"  query          : {s['query']}")
    print(f"  total hits     : {s['total_hits']}")
    print(f"  face-checked   : {s['face_checked']}")
    for m in s["matched_posts"]:
        print(f"  - {m.get('page_url', '')}  [score {m['match_score']}, {m['source']}, route={m.get('match_route','')}]")
    print()
    print("  ── blockchain ────────────────────────")
    bc = result["blockchain"]
    v = bc.get("verification")
    if v:
        print(f"  chain          : {bc['chain_name']}  height={bc['height']}")
        print(f"  block written  : #{bc['records'][0]['block_index']}")
        print(f"  data hash      : {bc['records'][0]['data_hash']}")
        print(f"  on-chain verify: {'VERIFIED' if v['verified'] else 'FAILED'}")
        print(f"  merkle root ok : {v['merkle_root_matches']}")
    else:
        print("  no record written (no matched post found)")
    print(f"  tamper check   : valid={bc['tamper_check']['valid']}  blocks={bc['tamper_check']['block_count']}")
    print(f"  elapsed        : {result['elapsed_seconds']}s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
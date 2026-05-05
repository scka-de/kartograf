from __future__ import annotations

import argparse
import json

from cartograph.mcp_servers import corpus_reader_github


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="psf")
    parser.add_argument("--repo", default="requests")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--state", default="closed")
    args = parser.parse_args()
    issues = corpus_reader_github.fetch_issues(args.owner, args.repo, args.limit, args.state)
    print(json.dumps({"count": len(issues), "mode": corpus_reader_github.get_mode()}))


if __name__ == "__main__":
    main()

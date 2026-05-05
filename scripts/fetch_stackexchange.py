from __future__ import annotations

import argparse
import json

from cartograph.mcp_servers import corpus_reader_stackexchange


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="money")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    examples = corpus_reader_stackexchange.fetch_questions(args.site, args.tag, args.limit)
    print(json.dumps({"count": len(examples), "mode": corpus_reader_stackexchange.get_mode()}))


if __name__ == "__main__":
    main()

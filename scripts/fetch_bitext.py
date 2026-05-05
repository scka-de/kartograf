from __future__ import annotations

import json
from pathlib import Path

from cartograph.mcp_servers import corpus_reader_bitext


def main() -> None:
    examples = corpus_reader_bitext.fetch_examples(limit=100_000)
    Path("data/corpora").mkdir(parents=True, exist_ok=True)
    print(json.dumps({"count": len(examples), "mode": corpus_reader_bitext.get_mode()}))


if __name__ == "__main__":
    main()

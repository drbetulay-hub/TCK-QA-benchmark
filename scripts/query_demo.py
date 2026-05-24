#!/usr/bin/env python3
"""Quick demo: one question via GraphRAG or BasicRAG."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="Hırsızlık suçunun cezası nedir?")
    parser.add_argument("--system", choices=["graphrag", "basicrag"], default="graphrag")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.system == "graphrag":
        from tck_graphrag.services.graphrag import QueryService

        svc = QueryService(provider=args.provider, model=args.model)
    else:
        from tck_graphrag.services.basic_rag import BasicRAGService

        svc = BasicRAGService(provider=args.provider, model=args.model)

    r = svc.query(args.question)
    print("Maddeler:", r.get("madde_sources"))
    print("\n", r["answer"][:2000])


if __name__ == "__main__":
    main()

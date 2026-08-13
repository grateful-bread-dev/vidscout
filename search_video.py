import argparse
import json
from pathlib import Path

import numpy as np

from src.embeddings import ClipSearchEngine


def main():
    parser = argparse.ArgumentParser(
        description="Search an indexed video using natural language."
    )
    parser.add_argument("query", help="Natural-language visual search query.")
    parser.add_argument(
        "--index",
        default="data/index",
        help="Index directory. Default: data/index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return. Default: 5",
    )
    args = parser.parse_args()

    index_dir = Path(args.index)

    embeddings = np.load(index_dir / "embeddings.npy")
    with open(index_dir / "metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)

    engine = ClipSearchEngine(model_name=metadata["model"])
    query_embedding = engine.embed_text(args.query)

    # Image and text embeddings are L2-normalized, so their dot product
    # is cosine similarity.
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][: args.top_k]

    print()
    print(f'Search results for: "{args.query}"')
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):
        frame = metadata["frames"][int(index)]
        print(
            f"{rank}. {frame['timestamp']}  "
            f"score={scores[index]:.4f}  "
            f"{frame['path']}"
        )


if __name__ == "__main__":
    main()

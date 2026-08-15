import argparse
import json
from pathlib import Path

import numpy as np

from src.embeddings import ClipSearchEngine


def aggregate_best_frame_per_shot(
    scores: np.ndarray,
    frames: list[dict],
) -> list[dict]:
    """
    For each shot, keep only the representative frame with
    the best similarity score.
    """
    best_by_shot: dict[int, dict] = {}

    for index, score in enumerate(scores):
        frame = frames[int(index)]
        shot_index = int(frame["shot_index"])

        current_best = best_by_shot.get(shot_index)

        if current_best is None or score > current_best["score"]:
            best_by_shot[shot_index] = {
                "shot_index": shot_index,
                "score": float(score),
                "frame": frame,
            }

    ranked = sorted(
        best_by_shot.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="Search an indexed video using natural language."
    )

    parser.add_argument(
        "query",
        help="Natural-language visual search query.",
    )

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

    # Embeddings are L2-normalized, so dot product == cosine similarity
    scores = embeddings @ query_embedding

    ranked_shots = aggregate_best_frame_per_shot(
        scores,
        metadata["frames"],
    )

    top_results = ranked_shots[: args.top_k]

    print()
    print(f'Search results for: "{args.query}"')
    print("=" * 70)

    for rank, result in enumerate(top_results, start=1):
        frame = result["frame"]
        shot_number = result["shot_index"] + 1

        print(
            f"{rank}. Shot {shot_number:03d}  "
            f"{frame['shot_start_timestamp']} -> "
            f"{frame['shot_end_timestamp']}  "
            f"score={result['score']:.4f}"
        )

        print(
            f"   Best frame: {frame['timestamp']}  "
            f"{frame['path']}"
        )


if __name__ == "__main__":
    main()
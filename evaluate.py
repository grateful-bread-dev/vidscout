import argparse
import json
from pathlib import Path

import numpy as np

from src.embeddings import ClipSearchEngine


# ---------------------------------------------------------
# Small built-in VidScout evaluation set
# ---------------------------------------------------------
#
# expected_match=True:
#   The query describes something genuinely present
#   in the soccer video.
#
# expected_match=False:
#   The query describes something absent from the video.
#
# Keeping this here instead of JSON is intentional for
# the current portfolio-sized evaluation.
# ---------------------------------------------------------

EVALUATION_QUERIES = [
    {
        "query": "a soccer ball",
        "expected_match": True,
    },
    {
        "query": "a man kicking a soccer ball",
        "expected_match": True,
    },
    {
        "query": "a man playing soccer",
        "expected_match": True,
    },
    {
        "query": "a person kicking a ball",
        "expected_match": True,
    },
    {
        "query": "a soccer player on a field",
        "expected_match": True,
    },
    {
        "query": "a person running on a soccer field",
        "expected_match": True,
    },

    # Negative queries: these concepts are not in the clip.
    {
        "query": "a woman holding a cup",
        "expected_match": False,
    },
    {
        "query": "a volcano erupting",
        "expected_match": False,
    },
    {
        "query": "an airplane landing on a runway",
        "expected_match": False,
    },
    {
        "query": "a dog swimming in a pool",
        "expected_match": False,
    },
    {
        "query": "a car driving at night",
        "expected_match": False,
    },
    {
        "query": "a person cooking in a kitchen",
        "expected_match": False,
    },
]


def aggregate_best_frame_per_shot(
    scores: np.ndarray,
    frames: list[dict],
) -> list[dict]:
    """
    Keep the highest-scoring representative frame
    for each detected shot.
    """

    best_by_shot = {}

    for frame_index, score in enumerate(scores):
        frame = frames[frame_index]

        shot_number = int(frame["shot_index"]) + 1

        current_best = best_by_shot.get(shot_number)

        if (
            current_best is None
            or score > current_best["score"]
        ):
            best_by_shot[shot_number] = {
                "shot_number": shot_number,
                "score": float(score),
                "frame": frame,
            }

    return sorted(
        best_by_shot.values(),
        key=lambda result: result["score"],
        reverse=True,
    )


def classification_accuracy(
    results: list[dict],
    threshold: float,
) -> float:
    """
    Treat a query as having a match when its highest
    similarity score is >= threshold.
    """

    correct = 0

    for result in results:
        predicted_match = (
            result["top_score"] >= threshold
        )

        if predicted_match == result["expected_match"]:
            correct += 1

    return correct / len(results)


def find_best_threshold(
    results: list[dict],
) -> tuple[float, float]:
    """
    Search candidate similarity thresholds and return
    the one that best separates positive and negative
    evaluation queries.

    This is calibration on a small evaluation set,
    not a universal CLIP confidence threshold.
    """

    scores = sorted(
        set(
            result["top_score"]
            for result in results
        )
    )

    if len(scores) == 1:
        return scores[0], classification_accuracy(
            results,
            scores[0],
        )

    candidates = []

    # Threshold slightly below all observed scores.
    candidates.append(scores[0] - 0.001)

    # Midpoints between every pair of neighboring scores.
    for lower, upper in zip(
        scores[:-1],
        scores[1:],
    ):
        candidates.append(
            (lower + upper) / 2.0
        )

    # Threshold slightly above all observed scores.
    candidates.append(scores[-1] + 0.001)

    best_threshold = candidates[0]
    best_accuracy = -1.0

    for threshold in candidates:
        accuracy = classification_accuracy(
            results,
            threshold,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VidScout semantic similarity "
            "using positive and negative queries."
        )
    )

    parser.add_argument(
        "--index",
        default="data/index",
        help="VidScout index directory. Default: data/index",
    )

    args = parser.parse_args()

    index_dir = Path(args.index)

    embeddings_path = (
        index_dir / "embeddings.npy"
    )

    metadata_path = (
        index_dir / "metadata.json"
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Could not find {embeddings_path}. "
            "Index a video first."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. "
            "Index a video first."
        )

    embeddings = np.load(
        embeddings_path
    )

    with open(
        metadata_path,
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    print("Loading CLIP once for evaluation...")

    engine = ClipSearchEngine(
        model_name=metadata["model"]
    )

    evaluation_results = []

    print()
    print("VidScout Evaluation")
    print("=" * 72)

    for item in EVALUATION_QUERIES:
        query = item["query"]
        expected_match = item["expected_match"]

        query_embedding = engine.embed_text(
            query
        )

        # Image and text embeddings are normalized,
        # so dot product gives cosine similarity.
        scores = embeddings @ query_embedding

        ranked_shots = aggregate_best_frame_per_shot(
            scores,
            metadata["frames"],
        )

        if not ranked_shots:
            raise RuntimeError(
                "The index contains no searchable shots."
            )

        best_result = ranked_shots[0]

        top_score = best_result["score"]

        second_score = (
            ranked_shots[1]["score"]
            if len(ranked_shots) > 1
            else None
        )

        evaluation_results.append(
            {
                "query": query,
                "expected_match": expected_match,
                "top_score": top_score,
            }
        )

        label = (
            "POSITIVE"
            if expected_match
            else "NEGATIVE"
        )

        print()
        print(
            f'[{label}] "{query}"'
        )

        print(
            f"Best shot: "
            f"{best_result['shot_number']:03d}"
        )

        print(
            f"Top similarity: "
            f"{top_score:.4f}"
        )

        if second_score is not None:
            print(
                f"Second similarity: "
                f"{second_score:.4f}"
            )

            print(
                f"Score margin: "
                f"{top_score - second_score:.4f}"
            )

        print(
            f"Best representative frame: "
            f"{best_result['frame']['path']}"
        )

    positive_scores = [
        result["top_score"]
        for result in evaluation_results
        if result["expected_match"]
    ]

    negative_scores = [
        result["top_score"]
        for result in evaluation_results
        if not result["expected_match"]
    ]

    best_threshold, calibration_accuracy = (
        find_best_threshold(
            evaluation_results
        )
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Queries evaluated: "
        f"{len(evaluation_results)}"
    )

    print(
        f"Positive queries: "
        f"{len(positive_scores)}"
    )

    print(
        f"Negative queries: "
        f"{len(negative_scores)}"
    )

    print()

    print(
        "Positive similarity range: "
        f"{min(positive_scores):.4f} "
        f"to {max(positive_scores):.4f}"
    )

    print(
        "Mean positive similarity: "
        f"{np.mean(positive_scores):.4f}"
    )

    print()

    print(
        "Negative similarity range: "
        f"{min(negative_scores):.4f} "
        f"to {max(negative_scores):.4f}"
    )

    print(
        "Mean negative similarity: "
        f"{np.mean(negative_scores):.4f}"
    )

    print()

    separation = (
        min(positive_scores)
        - max(negative_scores)
    )

    print(
        "Positive/negative separation: "
        f"{separation:.4f}"
    )

    print()

    print(
        "Best calibration threshold: "
        f"{best_threshold:.4f}"
    )

    print(
        "Calibration accuracy: "
        f"{calibration_accuracy:.1%}"
    )

    print()
    print(
        "Important: this threshold is calibrated "
        "on this small test set. It is not a CLIP "
        "confidence probability or a universal threshold."
    )


if __name__ == "__main__":
    main()
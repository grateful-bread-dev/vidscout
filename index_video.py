import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.embeddings import ClipSearchEngine
from src.shot_detection import detect_shots
from src.video import extract_representative_frames


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a shot-based semantic-search index for a video."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the source video.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help=(
            "Histogram-distance threshold for cut detection. "
            "Default: 0.45"
        ),
    )

    parser.add_argument(
        "--min-shot-seconds",
        type=float,
        default=0.5,
        help=(
            "Minimum shot duration in seconds. Default: 0.5"
        ),
    )

    parser.add_argument(
        "--output",
        default="data",
        help="Output directory. Default: data",
    )

    args = parser.parse_args()

    output_root = Path(args.output)
    frames_dir = output_root / "frames"
    index_dir = output_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Detecting shots...")
    shots, _ = detect_shots(
        video_path=args.video,
        threshold=args.threshold,
        min_shot_seconds=args.min_shot_seconds,
        debug=False,
    )
    print(f"Detected {len(shots)} shots.")

    print("Extracting representative frames...")
    frames, video_info = extract_representative_frames(
        args.video,
        shots,
        frames_dir,
    )
    print(f"Extracted {len(frames)} representative frames.")

    print("Generating CLIP image embeddings...")
    engine = ClipSearchEngine()
    embeddings = engine.embed_images(
        [frame.path for frame in frames]
    )

    np.save(index_dir / "embeddings.npy", embeddings)

    metadata = {
        "model": engine.model_name,
        "video": {
            **video_info,
            "shot_count": len(shots),
            "representative_frame_count": len(frames),
            "shot_detection_threshold": args.threshold,
            "min_shot_seconds": args.min_shot_seconds,
        },
        "shots": [asdict(shot) for shot in shots],
        "frames": [frame.to_dict() for frame in frames],
    }

    with open(
        index_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print()
    print("VidScout shot-based index created.")
    print(f"Detected shots: {len(shots)}")
    print(f"Representative frames: {len(frames)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Metadata: {index_dir / 'metadata.json'}")
    print(f"Embeddings: {index_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
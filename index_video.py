import argparse
import json
from pathlib import Path

import numpy as np

from src.embeddings import ClipSearchEngine
from src.video import sample_video


def main():
    parser = argparse.ArgumentParser(
        description="Create a semantic-search index for a video."
    )
    parser.add_argument("video", help="Path to the source video.")
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between sampled frames. Default: 3.0",
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

    print("Sampling video...")
    frames, video_info = sample_video(
        args.video,
        frames_dir,
        interval_seconds=args.interval,
    )
    print(f"Extracted {len(frames)} frames.")

    print("Generating CLIP image embeddings...")
    engine = ClipSearchEngine()
    embeddings = engine.embed_images([frame.path for frame in frames])

    np.save(index_dir / "embeddings.npy", embeddings)

    metadata = {
        "model": engine.model_name,
        "video": video_info,
        "frames": [frame.to_dict() for frame in frames],
    }

    with open(index_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print()
    print("VidScout index created.")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Metadata: {index_dir / 'metadata.json'}")
    print(f"Embeddings: {index_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    main()

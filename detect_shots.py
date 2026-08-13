import argparse

from src.shot_detection import (
    detect_shots,
    format_timestamp,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect editorial shot boundaries "
            "using histogram differences."
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
            "Histogram-distance threshold "
            "for detecting a cut. Default: 0.45"
        ),
    )

    parser.add_argument(
        "--min-shot-seconds",
        type=float,
        default=0.5,
        help=(
            "Minimum shot duration in seconds. "
            "Default: 0.5"
        ),
    )

    args = parser.parse_args()

    shots, fps = detect_shots(
        video_path=args.video,
        threshold=args.threshold,
        min_shot_seconds=args.min_shot_seconds,
        debug=True,
    )

    print()
    print("Shot detection complete.")
    print(f"FPS: {fps:.3f}")
    print(f"Detected shots: {len(shots)}")
    print()

    for shot in shots:
        start = format_timestamp(
            shot.start_seconds
        )

        end = format_timestamp(
            shot.end_seconds
        )

        print(
            f"Shot {shot.index + 1:03d}: "
            f"{start} -> {end} "
            f"(frames "
            f"{shot.start_frame}-"
            f"{shot.end_frame})"
        )


if __name__ == "__main__":
    main()
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
from PIL import Image


@dataclass
class SampledFrame:
    path: str
    seconds: float
    frame_number: int
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def sample_video(
    video_path: str | Path,
    output_dir: str | Path,
    interval_seconds: float = 3.0,
) -> tuple[list[SampledFrame], dict]:
    """
    Extract one representative frame every `interval_seconds`.

    This is intentionally simple for VidScout v0. Later we will replace
    fixed-interval sampling with real shot-boundary detection.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        capture.release()
        raise ValueError("Video reported an invalid frame rate.")

    duration_seconds = frame_count / fps
    frames: list[SampledFrame] = []

    sample_number = 0
    seconds = 0.0

    while seconds < duration_seconds:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        success, frame_bgr = capture.read()

        if success:
            actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            filename = f"frame_{sample_number:05d}.jpg"
            frame_path = output_dir / filename
            image.save(frame_path, quality=90)

            frames.append(
                SampledFrame(
                    path=str(frame_path),
                    seconds=seconds,
                    frame_number=actual_frame,
                    timestamp=format_timestamp(seconds),
                )
            )
            sample_number += 1

        seconds += interval_seconds

    capture.release()

    video_info = {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "interval_seconds": interval_seconds,
    }

    return frames, video_info

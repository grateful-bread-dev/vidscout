from dataclasses import dataclass, asdict
from pathlib import Path
import math

import cv2
from PIL import Image

from src.shot_detection import Shot


@dataclass
class RepresentativeFrame:
    path: str
    seconds: float
    frame_number: int
    timestamp: str
    shot_index: int
    shot_start_seconds: float
    shot_end_seconds: float
    shot_start_timestamp: str
    shot_end_timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def determine_sample_count(shot_duration_seconds: float) -> int:
    """
    Decide how many representative frames to extract for a shot.

    Rule:
    - shot <= 4 sec  -> 1 frame
    - shot <= 8 sec  -> 2 frames
    - shot > 8 sec   -> about 1 frame every ~4 seconds
    """
    if shot_duration_seconds <= 4.0:
        return 1

    if shot_duration_seconds <= 8.0:
        return 2

    return max(3, math.ceil(shot_duration_seconds / 4.0))


def choose_frame_numbers_for_shot(
    shot: Shot,
    fps: float,
) -> list[int]:
    """
    Pick representative frame numbers inside a shot.

    For multiple samples, we place them evenly across the shot
    and avoid choosing only the exact first/last frame.
    """
    shot_length_frames = max(
        1,
        shot.end_frame - shot.start_frame + 1,
    )

    shot_duration_seconds = shot_length_frames / fps

    sample_count = determine_sample_count(
        shot_duration_seconds
    )

    sample_count = min(sample_count, shot_length_frames)

    if sample_count == 1:
        midpoint = shot.start_frame + (
            shot_length_frames // 2
        )
        return [midpoint]

    candidate_frames = []

    for index in range(sample_count):
        fraction = (index + 1) / (sample_count + 1)

        relative_frame = round(
            fraction * (shot_length_frames - 1)
        )

        frame_number = shot.start_frame + relative_frame

        frame_number = max(
            shot.start_frame,
            min(shot.end_frame, frame_number),
        )

        candidate_frames.append(frame_number)

    # Deduplicate while preserving order
    deduped = []
    seen = set()

    for frame_number in candidate_frames:
        if frame_number not in seen:
            deduped.append(frame_number)
            seen.add(frame_number)

    return deduped


def extract_representative_frames(
    video_path: str | Path,
    shots: list[Shot],
    output_dir: str | Path,
) -> tuple[list[RepresentativeFrame], dict]:
    """
    Extract representative JPEG frames for every detected shot.
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

    representative_frames: list[RepresentativeFrame] = []

    for shot in shots:
        sample_frame_numbers = choose_frame_numbers_for_shot(
            shot,
            fps,
        )

        for sample_index, frame_number in enumerate(sample_frame_numbers):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame_bgr = capture.read()

            if not success:
                continue

            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(frame_rgb)

            filename = (
                f"shot_{shot.index:03d}_"
                f"sample_{sample_index:02d}.jpg"
            )

            frame_path = output_dir / filename
            image.save(frame_path, quality=90)

            seconds = frame_number / fps

            representative_frames.append(
                RepresentativeFrame(
                    path=str(frame_path),
                    seconds=seconds,
                    frame_number=frame_number,
                    timestamp=format_timestamp(seconds),
                    shot_index=shot.index,
                    shot_start_seconds=shot.start_seconds,
                    shot_end_seconds=shot.end_seconds,
                    shot_start_timestamp=format_timestamp(
                        shot.start_seconds
                    ),
                    shot_end_timestamp=format_timestamp(
                        shot.end_seconds
                    ),
                )
            )

    capture.release()

    video_info = {
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
    }

    return representative_frames, video_info

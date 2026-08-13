from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Shot:
    index: int
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float


def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)

    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{milliseconds:03d}"
    )


def compute_histogram(
    frame,
    resize_width: int = 320,
):
    """
    Convert a video frame into a normalized HSV color histogram.

    The histogram acts as a compact representation of the frame's
    overall color distribution.
    """

    height, width = frame.shape[:2]

    if width > resize_width:
        scale = resize_width / width

        resized_height = int(height * scale)

        frame = cv2.resize(
            frame,
            (resize_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    histogram = cv2.normalize(
        histogram,
        None,
        alpha=1.0,
        beta=0.0,
        norm_type=cv2.NORM_L1,
    )

    return histogram


def histogram_distance(
    histogram_a,
    histogram_b,
) -> float:
    """
    Compare two normalized histograms.

    Bhattacharyya distance is near 0 for very similar histograms
    and increases as the histograms become more different.
    """

    return float(
        cv2.compareHist(
            histogram_a,
            histogram_b,
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )


def detect_shots(
    video_path: str | Path,
    threshold: float = 0.45,
    min_shot_seconds: float = 0.5,
    debug: bool = False,
) -> tuple[list[Shot], float]:
    """
    Detect probable hard cuts in a video using consecutive-frame
    histogram differences.

    Parameters
    ----------
    video_path:
        Path to the source video.

    threshold:
        Histogram-distance threshold required to declare a cut.

    min_shot_seconds:
        Minimum allowed shot duration. This prevents rapid repeated
        detections from producing extremely short shots.

    debug:
        If True, prints every detected cut and its distance score.

    Returns
    -------
    shots:
        List of detected Shot objects.

    fps:
        Frame rate reported by OpenCV.
    """

    video_path = Path(video_path)

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        capture.release()

        raise ValueError(
            "Video reported an invalid frame rate."
        )

    minimum_shot_frames = max(
        1,
        round(min_shot_seconds * fps),
    )

    success, previous_frame = capture.read()

    if not success:
        capture.release()

        raise ValueError(
            "Could not read the first video frame."
        )

    previous_histogram = compute_histogram(
        previous_frame
    )

    current_shot_start = 0
    frame_index = 1

    shots: list[Shot] = []

    while True:
        success, frame = capture.read()

        if not success:
            break

        current_histogram = compute_histogram(
            frame
        )

        distance = histogram_distance(
            previous_histogram,
            current_histogram,
        )

        shot_length = (
            frame_index - current_shot_start
        )

        enough_frames = (
            shot_length >= minimum_shot_frames
        )

        if (
            distance >= threshold
            and enough_frames
        ):
            shot_index = len(shots)

            shots.append(
                Shot(
                    index=shot_index,
                    start_frame=current_shot_start,
                    end_frame=frame_index - 1,
                    start_seconds=(
                        current_shot_start / fps
                    ),
                    end_seconds=(
                        frame_index / fps
                    ),
                )
            )

            if debug:
                print(
                    f"Cut detected at "
                    f"{format_timestamp(frame_index / fps)} "
                    f"(frame {frame_index}, "
                    f"distance={distance:.4f})"
                )

            current_shot_start = frame_index

        previous_histogram = current_histogram
        frame_index += 1

    capture.release()

    if frame_index > current_shot_start:
        shots.append(
            Shot(
                index=len(shots),
                start_frame=current_shot_start,
                end_frame=frame_index - 1,
                start_seconds=(
                    current_shot_start / fps
                ),
                end_seconds=(
                    frame_index / fps
                ),
            )
        )

    return shots, fps
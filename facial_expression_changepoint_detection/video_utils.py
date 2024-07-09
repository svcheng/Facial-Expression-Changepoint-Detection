import os
from pathlib import Path
from typing import Generator

import cv2 as cv
import numpy as np


def get_frames(vid_path: Path) -> Generator[tuple[np.ndarray, float], None, int]:
    """
    Returns:
        An iterator through the frames of the video
    """

    video = cv.VideoCapture(str(vid_path))
    if not video.isOpened():
        return -1

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        timestamp = video.get(cv.CAP_PROP_POS_MSEC)
        yield frame, timestamp

    video.release()
    return 0


def save_frames(
    output_dir: Path, frames: list[np.ndarray], filenames: list[str]
) -> None:
    """
    Saves the given frames in the given directory. Raises error if the given path does not exist, or exists but points to a file.
    """

    if not output_dir.exists():
        raise ValueError("Given path does not exist.")
    if output_dir.is_file():
        raise ValueError("Given path should be a path to a directoy, not a file.")

    cwd = Path.cwd()
    os.chdir(output_dir)
    for frame, filename in zip(frames, filenames):
        cv.imwrite(filename=filename, img=frame)
    os.chdir(cwd)

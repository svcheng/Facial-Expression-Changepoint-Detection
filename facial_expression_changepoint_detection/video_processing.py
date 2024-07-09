from abc import abstractmethod
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
import ruptures as rpt
from scipy.signal import savgol_filter

from .landmarks import LandmarksSignalExtractor
from .video_utils import get_frames, save_frames


class SignalExtractor(Protocol):
    """A protocol for classes that can extract signals from videos"""

    @abstractmethod
    def extract_signal(self, vid_path: Path) -> np.ndarray: ...


class VideoProcessor:
    def __init__(
        self,
        vid_path: Path,
        signal_extractor: Optional[SignalExtractor] = None,
        noise_filterer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        changepoint_detector: Optional[rpt.BottomUp] = None,
    ) -> None:
        self.vid_path = vid_path
        self.signal_extractor = (
            LandmarksSignalExtractor() if signal_extractor is None else signal_extractor
        )

        def default_filter(signal: np.ndarray):
            """Savitsky-Golay filter with window length of 17 and polynomial order of 13"""

            return savgol_filter(
                signal, window_length=17, polyorder=13, axis=-1, mode="nearest"
            )

        self.noise_filterer = (
            default_filter if noise_filterer is None else noise_filterer
        )
        self.changepoint_detector = (
            rpt.BottomUp(model="rbf", jump=1)  # bottom-up change detection algorithm
            if changepoint_detector is None
            else changepoint_detector
        )
        self.filtered_signal = None

    def get_changepoints(self, num_changepoints: int) -> list[int]:
        """
        Computes the changepoints of the video by extracting a signal from it using the instance's SignalExtractor,
        filters it using the instance's noise filtering function, then computes the changepoints from that filtered signal.
        Stores the filtered signal the first time it is called (regardless of what value is passed as num_changepoints)
        to avoid having to recompute it.

        Returns:
            The computed changepoints
        """

        if self.filtered_signal is None:
            signal = self.signal_extractor.extract_signal(self.vid_path)
            self.filtered_signal = self.noise_filterer(signal)

        return self.changepoint_detector.fit_predict(
            signal=self.filtered_signal, n_bkps=num_changepoints
        )[:-1]

    def select_frames(self, frame_count: int) -> list[np.ndarray]:
        """
        Returns:
            The frame_count frames at the detected changepoints
        """

        changepoints = self.get_changepoints(num_changepoints=frame_count)
        frames = [
            frame
            for i, (frame, _) in enumerate(get_frames(vid_path=self.vid_path))
            if i in changepoints
        ]
        return frames

    def select_and_save_frames(self, frame_count: int, output_dir: Path) -> None:
        """
        Selects frames from the video then saves them in the specified path
        """

        frames = self.select_frames(frame_count=frame_count)
        filenames = [f"{self.vid_path.stem}_{i}.png" for i in range(frame_count)]
        save_frames(output_dir=output_dir, frames=frames, filenames=filenames)

    def process(self, frame_counts: list[int], output_dir: Path) -> None:
        """
        Calls the select_and_save_frames method for all the frame_counts, saving files in the following directory structure:
        output_dir/
            frame_counts[0]/
            frame_counts[1]/
            .
            .
            .
        """

        for frame_count in frame_counts:
            subdir = output_dir / str(frame_count)
            if not subdir.exists():
                Path.mkdir(subdir, parents=True)
            self.select_and_save_frames(frame_count=frame_count, output_dir=subdir)

import itertools
import random
import time
from multiprocessing import Pool
from pathlib import Path

from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.visualization import Animation

# ==================================== CONFIGS ====================================

DATASET_PATH = Path(__file__).parent.parent / "DAiSEE" / "DataSet"
SUBFOLDERS = [DATASET_PATH / subfolder for subfolder in ["Train", "Validation", "Test"]]
FRAME_COUNTS = [1, 2, 3]
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def get_all_videos() -> list[Path]:
    return [
        Path(dirpath) / filename
        for dirpath, _, filenames in itertools.chain(
            *[subfolder.walk(on_error=print) for subfolder in SUBFOLDERS]
        )
        for filename in filenames
    ]


def visualize(vid_paths: list[Path], frame_count: int) -> None:
    for vid_path in vid_paths:
        video_processor = VideoProcessor(
            vid_path=vid_path,
            signal_extractor=LandmarksSignalExtractor(indices={20}),
        )
        animation = Animation(
            vid_path,
            video_processor=video_processor,
            frame_count=frame_count,
            title=f"Video ID: {vid_path.stem}",
        )
        animation.run()


def process_video(vid_path: Path) -> str:
    """
    Processes a single video. Declared globally so that it can be passed to a multiprocessing pool

    Returns:
        The filename of the video passed
    """

    vp = VideoProcessor(vid_path=vid_path)
    vp.process(frame_counts=FRAME_COUNTS, output_dir=OUTPUT_DIR)
    return str(vid_path.name)


def run(vid_paths: list[Path], chunksize: int = 8) -> None:
    """
    Processes the given videos, using multiprocessing to speed up execution
    """

    with Pool() as pool:
        filenames = pool.imap_unordered(process_video, vid_paths, chunksize=chunksize)
        for filename in filenames:
            print(f"Finished processing of video {filename}\n")


def benchmark(sample_size: int, chunksizes: list[int]) -> None:
    """
    Compares performance of the program on a random sample of all the videos when running with no multiprocessing, and with various chunk sizes for multiprocessing
    """

    random.seed(0)
    vid_paths = random.sample(get_all_videos(), k=sample_size)
    num_chunksizes = len(chunksizes)
    multiprocessing_times = [0.0] * num_chunksizes

    # time with no multiprocessing
    t0 = time.perf_counter()
    for vid in vid_paths:
        filename = process_video(vid)
        print(f"Finished processing of video: {filename}\n")
    elapsed_single = time.perf_counter() - t0

    # time with multiprocessing
    for i in range(num_chunksizes):
        t0 = time.perf_counter()
        run(vid_paths=vid_paths, chunksize=chunksizes[i])
        multiprocessing_times[i] = time.perf_counter() - t0

    # display results
    print("Time Elapsed (in seconds):")
    print(f"    No multiprocessing: {elapsed_single}")
    for chunksize, elapsed_time in zip(chunksizes, multiprocessing_times):
        print(f"    Multiprocessing chunksize={chunksize}: {elapsed_time}")

    """
    with 32 videos
    Time Elapsed (in seconds):
        No multiprocessing: 149.74174900026992
        Multiprocessing chunksize=1: 90.34611150017008
        Multiprocessing chunksize=4: 103.70775760011747
        Multiprocessing chunksize=8: 87.80353360017762
        Multiprocessing chunksize=16: 99.86084290035069
    """


def main() -> None:
    all_vids = get_all_videos()

    random.seed(0)
    all_vids = random.sample(all_vids, k=2)
    t0 = time.perf_counter()
    run(vid_paths=all_vids)
    print(
        f"Finished processing all videos in {(time.perf_counter() - t0)/(60*60)} hours."
    )


if __name__ == "__main__":
    main()

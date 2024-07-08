import itertools
import random
import time
from multiprocessing import Pool
from pathlib import Path

from facial_expression_changepoint_detection.landmarks import LandmarksSignalExtractor
from facial_expression_changepoint_detection.video_processing import VideoProcessor
from facial_expression_changepoint_detection.visualization import Animation

# ================== CONFIGS ==================

DATASET_PATH = Path(__file__).parent.parent / "DAiSEE" / "DataSet"
SUBFOLDERS = [DATASET_PATH / subfolder for subfolder in ["Train", "Validation", "Test"]]
EXAMPLE_VIDS = [
    SUBFOLDERS[0] / vid
    for vid in [
        # boredom, engagement, confusion, frustration
        "303830/303830274/303830274.mp4",  # 0 3 1 1
        "310068/3100681045/3100681045.avi",  # 2 2 1 1
        "310081/3100811002/3100811002.avi",  # 0 2 0 0
        "400018/4000182069/4000182069.avi",  # 0 2 0 0
        "210058/2100582062/2100582062.avi",  # 3 0 0 0
        "110002/1100022056/1100022056.avi",  # 0 3 0 0
        "111003/1110032004/1110032004.avi",  # 1 2 3 1
        "210057/2100571044/2100571044.avi",  # 2 1 0 3
    ]
]
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
    vp = VideoProcessor(vid_path=vid_path)
    vp.process(frame_counts=FRAME_COUNTS, output_dir=OUTPUT_DIR)
    return str(vid_path.name)


def run(vid_paths: list[Path], chunksize: int = 8) -> None:
    with Pool() as pool:
        filenames = pool.imap_unordered(process_video, vid_paths, chunksize=chunksize)
        for filename in filenames:
            print(f"Finished processing of video {filename}\n")


def benchmark(sample_size: int, chunksizes: list[int]) -> None:
    """Compares performance of the program on a random sample of all the videos with no multiprocessing, and with various chunk sizes for multiprocessing."""

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
    run(vid_paths=all_vids[:3])


if __name__ == "__main__":
    main()

import cv2
import os
import re

from pathlib import Path


def video_writer_same_codec(video: cv2.VideoCapture, save_path: str) -> cv2.VideoWriter:
    """
    This function checks whether the directory of `save_path` exists and creates it if it's not.
    If `save_path` is an empty string, the video will not be saved.

    Returns: a VideoWriter object with the same codec as the input VideoCapture object.
    """

    splitted = re.split('\\\|/', save_path)
    dir_path = "/".join(splitted[:-1])

    if save_path != "" and not os.path.exists(dir_path):
        path = Path(dir_path)
        path.mkdir(parents=True)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"avc1")
    return cv2.VideoWriter(save_path, codec, fps, (w, h))
import cv2


def video_writer_same_codec(video: cv2.VideoCapture, save_path: str) -> cv2.VideoWriter:
    """
    This function returns a VideoWriter object with the same codec as the input VideoCapture object
    """
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"avc1")
    return cv2.VideoWriter(save_path, codec, fps, (w, h))